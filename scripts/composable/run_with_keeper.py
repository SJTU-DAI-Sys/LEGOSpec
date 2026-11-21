"""Run one node's training with the requested keeper lifecycle."""
import argparse
import fcntl
import hashlib
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


def matching_keepers(script):
    """Match the executed Python/shell script, never a helper's option value."""
    found = []
    proc = Path('/proc')
    if not proc.exists():
        return found
    for directory in proc.iterdir():
        if not directory.name.isdigit():
            continue
        try:
            argv = (directory / 'cmdline').read_bytes().decode(errors='replace').split('\0')
            interpreter = Path(argv[0]).name if argv else ''
            if 'python' not in interpreter and interpreter not in ('bash', 'sh', 'zsh', 'dash'):
                continue
            index = 1
            while index < len(argv):
                argument = argv[index]
                if argument in ('-c', '-m'):
                    break
                if argument in ('-W', '-X'):
                    index += 2
                    continue
                if argument == '--':
                    index += 1
                    if index >= len(argv):
                        break
                    argument = argv[index]
                elif argument.startswith('-'):
                    index += 1
                    continue
                # Only the executed script counts, never a helper's option value.
                candidate = Path(argument)
                if candidate.name == script.name:
                    if not candidate.is_absolute():
                        candidate = directory / 'cwd' / candidate
                    if candidate.resolve() == script:
                        found.append(int(directory.name))
                break
        except (OSError, ValueError):
            continue
    return found


def process_state(pid):
    try:
        fields = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()
        return fields[0], int(fields[1]), int(fields[2]), fields[19]
    except (OSError, IndexError, ValueError):
        return None


def stop_keepers(script, launcher):
    schedulers = matching_keepers(launcher)
    targets = set(schedulers + matching_keepers(script))
    # Capture timeout/sleep children before terminating their scheduler. This
    # also handles legacy launchers that did not create their own process group.
    states = {}
    children = {}
    proc = Path('/proc')
    if proc.exists():
        for entry in proc.iterdir():
            if entry.name.isdigit():
                pid = int(entry.name)
                state = process_state(pid)
                if state is not None:
                    states[pid] = state
                    children.setdefault(state[1], []).append(pid)
    pending = list(schedulers)
    while pending:
        for child in children.get(pending.pop(), []):
            if child not in targets:
                targets.add(child)
                pending.append(child)
    identities = {pid: states[pid][3] for pid in targets if pid in states}
    groups = [pid for pid in schedulers if pid in states and states[pid][2] == pid]

    def alive(pid):
        state = process_state(pid)
        return state is not None and state[0] != 'Z' and state[3] == identities[pid]

    def send(number):
        for group in groups:
            try:
                os.killpg(group, number)
            except ProcessLookupError:
                pass
        for pid in identities:
            if alive(pid):
                try:
                    os.kill(pid, number)
                except ProcessLookupError:
                    pass

    send(signal.SIGTERM)
    deadline = time.monotonic() + 10
    while any(alive(pid) for pid in identities):
        if time.monotonic() >= deadline:
            send(signal.SIGKILL)
            break
        time.sleep(0.1)
    # Remove any late child in the scheduler's private process group as well.
    send(signal.SIGKILL)
    deadline = time.monotonic() + 5
    while any(alive(pid) for pid in identities):
        if time.monotonic() >= deadline:
            raise RuntimeError('Keeper processes did not exit after SIGKILL')
        time.sleep(0.1)


def terminate_group(process):
    if process is None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait()
    # Also remove children that outlived the launcher itself.
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stop-script', type=Path)
    parser.add_argument('--keeper-script', type=Path, required=True)
    parser.add_argument('--keeper-launcher', type=Path, required=True,
                        help='Original periodic shell launcher (keep.sh)')
    parser.add_argument('--keeper-log', type=Path)
    parser.add_argument('--stop-only', action='store_true', help='Stop all instances of the keeper script')
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    for file in (args.keeper_script, args.keeper_launcher):
        if not file.is_file():
            parser.error(f'missing lifecycle script: {file}')
    args.keeper_script = args.keeper_script.resolve()
    args.keeper_launcher = args.keeper_launcher.resolve()
    if args.stop_only:
        stop_keepers(args.keeper_script, args.keeper_launcher)
        print('[keeper] periodic scheduler and all keeper workers stopped', flush=True)
        return 0
    command = args.command[1:] if args.command[:1] == ['--'] else args.command
    if not command:
        parser.error('training command is required')
    if args.stop_script is None or not args.stop_script.is_file() or args.keeper_log is None:
        parser.error('training requires an existing --stop-script and --keeper-log')
    # A local per-host lock prevents a second launcher from restarting the keeper
    # while this node is still training. The file contains no credentials.
    identity = hashlib.sha256(str(args.keeper_script).encode()).hexdigest()[:16]
    lock = open(f'/tmp/legospec_keeper_{os.getuid()}_{identity}.lock', 'a')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        parser.error('another training launcher already manages this node keeper')

    process = None
    interrupted = 0
    restore = False

    def on_signal(number, frame):
        nonlocal interrupted
        interrupted = number
        if process is not None:
            try:
                os.killpg(process.pid, number)
            except ProcessLookupError:
                pass

    signal.signal(signal.SIGTERM, on_signal)
    signal.signal(signal.SIGINT, on_signal)
    result = 1
    try:
        print(f'[keeper] stopping via {args.stop_script}', flush=True)
        restore = True
        stopped = subprocess.run(['bash', str(args.stop_script)], check=False)
        stop_keepers(args.keeper_script, args.keeper_launcher)
        if stopped.returncode:
            print(f'[keeper] stop script failed: {stopped.returncode}', file=sys.stderr, flush=True)
            result = stopped.returncode
        elif interrupted:
            result = 128 + interrupted
        else:
            print('[keeper] starting training', flush=True)
            process = subprocess.Popen(command, start_new_session=True)
            while process.poll() is None:
                if interrupted:
                    terminate_group(process)
                    break
                time.sleep(0.1)
            result = process.wait()
            result = 128 + interrupted if interrupted else (128 - result if result < 0 else result)
    finally:
        # Do not restore keeper until all workers from this launch are stopped.
        terminate_group(process)
        if restore:
            stop_keepers(args.keeper_script, args.keeper_launcher)
            args.keeper_log.parent.mkdir(parents=True, exist_ok=True)
            env = dict(os.environ)
            env['PATH'] = str(Path(sys.executable).parent) + os.pathsep + env.get('PATH', '')
            with args.keeper_log.open('ab') as log:
                keeper = subprocess.Popen(['bash', str(args.keeper_launcher)],
                    stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                    env=env, start_new_session=True, close_fds=True)
            print(f'[keeper] periodic launcher restarted PID={keeper.pid} '
                  f'script={args.keeper_launcher} log={args.keeper_log}', flush=True)
        lock.close()
    return result


if __name__ == '__main__':
    raise SystemExit(main())
