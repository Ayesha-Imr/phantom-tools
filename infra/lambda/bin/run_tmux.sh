#!/usr/bin/env bash
# Resilient variant of bin/run.sh: the experiment runs inside a detached tmux
# session on the pod (survives laptop<->pod network blips), its output goes to
# runlogs/run.log on the persistent filesystem, and this script polls that log
# over fresh short-lived SSH connections until a completion marker appears,
# streaming new log lines as they arrive. The pod is stopped by the exit trap
# exactly like run.sh.
#
# Same args as run.sh: --repo, --project-name, --branch, --cmd, --req-file,
# --lock-file, --instance-type, --pod-id, --keep-alive, --yes.
# Note: --cmd must not contain double quotes (it is sent via tmux send-keys).

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LAMBDA_CLI="python3 $HERE/lambda/lambda_cli.py"

REPO=""
PROJECT_NAME=""
BRANCH="main"
CMD=""
REQ_FILE="requirements.txt"
LOCK_FILE="requirements.lock.txt"
INSTANCE_TYPE_OVERRIDE=""
POD_ID=""
KEEP_ALIVE=0
CONFIRM=0

while [ $# -gt 0 ]; do
    case "$1" in
        --repo) REPO="$2"; shift 2 ;;
        --project-name) PROJECT_NAME="$2"; shift 2 ;;
        --branch) BRANCH="$2"; shift 2 ;;
        --cmd) CMD="$2"; shift 2 ;;
        --req-file) REQ_FILE="$2"; shift 2 ;;
        --lock-file) LOCK_FILE="$2"; shift 2 ;;
        --instance-type) INSTANCE_TYPE_OVERRIDE="$2"; shift 2 ;;
        --pod-id) POD_ID="$2"; shift 2 ;;
        --keep-alive) KEEP_ALIVE=1; shift ;;
        --yes) CONFIRM=1; shift ;;
        -h|--help) grep '^#' "$0" | sed 's/^# \?//'; exit 0 ;;
        *) echo "Unknown arg: $1" >&2; exit 1 ;;
    esac
done

[ -n "$REPO" ] || { echo "ERROR: --repo is required" >&2; exit 1; }
[ -n "$CMD" ] || { echo "ERROR: --cmd is required" >&2; exit 1; }
[ -z "$PROJECT_NAME" ] && PROJECT_NAME="$(basename "$REPO" .git)"

eval "$(python3 "$HERE/lambda/config_to_env.py")"
SSH_PUBLIC_KEY_FILE="${SSH_PUBLIC_KEY_FILE/#\~/$HOME}"
SSH_PRIVATE_KEY_FILE="${SSH_PRIVATE_KEY_FILE/#\~/$HOME}"
SSH_USER="${SSH_USER:-ubuntu}"

[ -n "$FILESYSTEM" ] && [ -n "$REGION" ] || {
    echo "ERROR: lambda/config.yaml has no filesystem/region set yet." >&2; exit 1; }

echo "== GPU pod run (tmux-resilient) =="
echo "Repo:         $REPO ($BRANCH)"
echo "Project:      $PROJECT_NAME"
echo "Filesystem:   $FILESYSTEM  (region $REGION)"
echo "Command:      $(printf '%s' "$CMD" | sed -E 's/((HF_TOKEN|HUGGINGFACE_TOKEN|OPENAI_API_KEY|GEMINI_API_KEY|LAMBDA_API_KEY)=)[^ ]*/\1<redacted>/g')"
echo

if ! ssh-add -l 2>/dev/null | grep -q "$(ssh-keygen -lf "$SSH_PUBLIC_KEY_FILE" | awk '{print $2}')"; then
    echo "-- Loading $SSH_PRIVATE_KEY_FILE into ssh-agent --"
    ssh-add "$SSH_PRIVATE_KEY_FILE"
fi

INSTANCE_ID=""
WE_LAUNCHED=0

cleanup() {
    if [ "$WE_LAUNCHED" = "1" ] && [ "$KEEP_ALIVE" != "1" ] && [ -n "$INSTANCE_ID" ]; then
        echo
        echo "-- Stopping pod $INSTANCE_ID (cleanup) --"
        $LAMBDA_CLI stop --id "$INSTANCE_ID" || echo "WARNING: stop failed - check manually with:  $LAMBDA_CLI list" >&2
    elif [ -n "$INSTANCE_ID" ] && [ "$KEEP_ALIVE" = "1" ]; then
        echo
        echo "!!! --keep-alive was set: pod $INSTANCE_ID is STILL RUNNING and billing. !!!"
        echo "!!! Stop it yourself when done:                                            !!!"
        echo "!!!   python3 $HERE/lambda/lambda_cli.py stop --id $INSTANCE_ID            !!!"
    fi
}
trap cleanup EXIT INT TERM

if [ -n "$POD_ID" ]; then
    INSTANCE_ID="$POD_ID"
    echo "-- Reusing existing pod $INSTANCE_ID --"
else
    if [ "$CONFIRM" != "1" ]; then
        echo "Refusing to launch a new pod without --yes." >&2
        echo "Get the human's confirmation in chat first, then re-run with --yes." >&2
        exit 2
    fi
    if [ -n "$INSTANCE_TYPE_OVERRIDE" ]; then
        INSTANCE_TYPE="$INSTANCE_TYPE_OVERRIDE"
    else
        echo "-- Finding an instance type with live capacity in $REGION --"
        read -r INSTANCE_TYPE _ < <($LAMBDA_CLI find-capacity --region "$REGION" --prefer "${PREFERRED_INSTANCE_TYPES[@]}")
    fi
    echo "-- Launching $INSTANCE_TYPE in $REGION --"
    INSTANCE_ID=$($LAMBDA_CLI launch --instance-type "$INSTANCE_TYPE" --region "$REGION" \
        --ssh-key "$SSH_KEY_NAME" --filesystem "$FILESYSTEM" \
        --name "${PROJECT_NAME}-agent-run" --yes | tail -1)
    WE_LAUNCHED=1
    echo "Launched instance: $INSTANCE_ID"
fi

echo "-- Waiting for SSH --"
IP=$($LAMBDA_CLI wait-ssh --id "$INSTANCE_ID" --timeout 600)
echo "Instance reachable at $IP"

# Lambda recycles public IPs between tenants, so a fresh pod routinely lands on an
# IP already in known_hosts under a previous instance's key. Drop any stale entry.
ssh-keygen -R "$IP" >/dev/null 2>&1 || true

SSH_OPTS=(-i "$SSH_PRIVATE_KEY_FILE" -o StrictHostKeyChecking=accept-new -A)

POD_WORKSPACE="/lambda/nfs/$FILESYSTEM"
REMOTE_REPO_DIR="$POD_WORKSPACE/repos/$PROJECT_NAME"
ENV_FILE="$POD_WORKSPACE/envs/${PROJECT_NAME}.env"
RUNLOG="$REMOTE_REPO_DIR/runlogs/run.log"
DONE="$REMOTE_REPO_DIR/runlogs/run_done.txt"

echo "-- Syncing repo on pod --"
ssh "${SSH_OPTS[@]}" "$SSH_USER@$IP" bash -s <<EOF
set -euo pipefail
mkdir -p "$POD_WORKSPACE/repos"
export GIT_SSH_COMMAND="ssh -o StrictHostKeyChecking=accept-new"
# A concurrent pod may be cloning the same dir on the shared filesystem; wait it out.
for attempt in 1 2 3 4 5 6 7 8 9 10 11 12; do
    if [ -d "$REMOTE_REPO_DIR/.git" ] && [ -e "$REMOTE_REPO_DIR/.git/HEAD" ]; then
        break
    fi
    sleep 5
done
if [ -d "$REMOTE_REPO_DIR/.git" ] && [ -e "$REMOTE_REPO_DIR/.git/HEAD" ]; then
    cd "$REMOTE_REPO_DIR"
    git fetch origin
    git checkout "$BRANCH"
    # A persistent filesystem can retain pod-local WIP from an interrupted run.
    # Preserve it before syncing; never make the next run fail or discard it.
    # NB: the command substitutions are escaped so they evaluate ON the pod
    # (the heredoc is unquoted and would otherwise run them locally).
    if ! git diff --quiet || [ -n "\$(git status --porcelain --untracked-files=all)" ]; then
        git stash push --include-untracked -m "pre-run pod WIP \$(date -u +%Y%m%dT%H%M%SZ)"
    fi
    git pull --ff-only origin "$BRANCH"
else
    git clone --branch "$BRANCH" "$REPO" "$REMOTE_REPO_DIR"
fi
EOF

echo "-- Bootstrapping env --"
ssh "${SSH_OPTS[@]}" "$SSH_USER@$IP" \
    "WORKSPACE_DIR='$POD_WORKSPACE' PROJECT_NAME='$PROJECT_NAME' REPO_DIR='$REMOTE_REPO_DIR' REQ_FILE='$REQ_FILE' LOCK_FILE='$LOCK_FILE' bash -s" \
    < "$HERE/pod-env/bootstrap_pod.sh"

echo "-- Starting experiment inside detached tmux on the pod --"
ssh "${SSH_OPTS[@]}" "$SSH_USER@$IP" bash -s <<EOF
set -euo pipefail
cd "$REMOTE_REPO_DIR"
mkdir -p runlogs
rm -f "$DONE"
tmux kill-session -t smokerun 2>/dev/null || true
tmux new-session -d -s smokerun
tmux send-keys -t smokerun "source '$ENV_FILE'; cd '$REMOTE_REPO_DIR'; ($CMD) > runlogs/run.log 2>&1; echo exit_\$? > runlogs/run_done.txt" Enter
sleep 2
tmux ls
EOF

START_TS=$(date +%s)
POLL_SECS=$(( ${MAX_RUNTIME_HOURS:-3} * 3600 ))
[ "$POLL_SECS" -gt 0 ] || POLL_SECS=10800
elapsed=0
lines=0
DONE_FLAG=0
echo "-- Polling run.log (new lines only) --"
while [ "$elapsed" -lt "$POLL_SECS" ]; do
    sleep 15
    elapsed=$((elapsed + 15))
    if ssh "${SSH_OPTS[@]}" "$SSH_USER@$IP" "test -f '$DONE'" 2>/dev/null; then
        DONE_FLAG=1
        break
    fi
    new=$(ssh "${SSH_OPTS[@]}" "$SSH_USER@$IP" "wc -l < '$RUNLOG'" 2>/dev/null || echo 0)
    if [ "$new" -gt "$lines" ] 2>/dev/null; then
        ssh "${SSH_OPTS[@]}" "$SSH_USER@$IP" "tail -n +$((lines + 1)) '$RUNLOG'" 2>/dev/null || true
        lines=$new
    fi
done

echo
echo "-- Run finished (or poll budget exhausted) --"
ssh "${SSH_OPTS[@]}" "$SSH_USER@$IP" "cat '$DONE' 2>/dev/null || echo 'no completion marker'; echo '--- last log lines ---'; tail -20 '$RUNLOG' 2>/dev/null || true" || true

END_TS=$(date +%s)
echo "== done in ~$(( (END_TS - START_TS) / 60 )) min (excludes launch/boot wait) =="
# cleanup trap handles stopping the pod.
