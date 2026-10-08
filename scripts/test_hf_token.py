"""CPU checks for passing a Hugging Face download token by variable or file."""
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from clef_service.access import configured_hf_token

TOKEN = 'hf_testTokenThatMustNeverBeLogged123'


def expect_error(env, message):
    try:
        configured_hf_token(env)
    except ValueError as exc:
        assert message in str(exc), exc
        assert TOKEN not in str(exc)
    else:
        raise AssertionError(f'expected error: {message}')


with tempfile.TemporaryDirectory() as folder:
    good, empty, spaced = (Path(folder) / name for name in ('token', 'empty', 'spaced'))
    good.write_text(TOKEN + '\n'); empty.write_text(''); spaced.write_text('hf_with space\n')

    assert configured_hf_token({}) == ''
    assert configured_hf_token({'HF_TOKEN': TOKEN}) == TOKEN
    assert configured_hf_token({'HF_TOKEN_FILE': str(good)}) == TOKEN
    # Compose passes an empty HF_TOKEN alongside a file; that is not a conflict.
    assert configured_hf_token({'HF_TOKEN': '', 'HF_TOKEN_FILE': str(good)}) == TOKEN
    expect_error({'HF_TOKEN': TOKEN, 'HF_TOKEN_FILE': str(good)}, 'Set only one')
    expect_error({'HF_TOKEN_FILE': str(empty)}, 'file is empty')
    expect_error({'HF_TOKEN_FILE': str(Path(folder) / 'missing')}, 'Cannot read')
    expect_error({'HF_TOKEN_FILE': str(spaced)}, 'printable ASCII')

    # The launcher rejects a bad token file before touching the GPU or network,
    # and never echoes a token in its error.
    env = {**os.environ, 'HF_TOKEN': TOKEN, 'PYTHONPATH': str(ROOT)}
    result = subprocess.run([sys.executable, '-m', 'clef_service', 'download', '--hf-token-file', str(empty)],
                            capture_output=True, text=True, env=env, cwd=ROOT, timeout=120)
    assert result.returncode == 2, (result.returncode, result.stderr[-500:])
    assert 'Hugging Face token file is empty' in result.stderr, result.stderr[-500:]
    assert TOKEN not in result.stdout + result.stderr

print('PASS: HF_TOKEN and HF_TOKEN_FILE, empty Compose variable, conflicts, empty/unreadable/invalid files, CLI file precedence and no token in errors')
