"""Send explicitly requested experiment notifications through the existing local hook."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import subprocess


def notify(message: str, run_dir: Path, *, event='complete', evidence: Path | None = None) -> dict:
    run_dir = Path(run_dir)
    receipt_path = run_dir / 'DISCORD_NOTIFICATION.json'
    evidence_sha = hashlib.sha256(Path(evidence).read_bytes()).hexdigest() if evidence else None
    key = hashlib.sha256(json.dumps([event, evidence_sha or message]).encode()).hexdigest()
    if receipt_path.exists():
        previous = json.loads(receipt_path.read_text())
        if previous.get('status') == 'sent' and previous.get('deduplication_key') == key:
            print('Discord notification already sent for these results.', flush=True)
            return previous
    hook = Path.home() / '.claude/hooks/discord-notify.sh'
    receipt = {
        'event': event, 'message': message, 'deduplication_key': key,
        'evidence': str(evidence) if evidence else None, 'evidence_sha256': evidence_sha,
        'attempted_at_utc': datetime.now(timezone.utc).isoformat(), 'hook': str(hook),
    }
    try:
        # timeout also terminates the hook's curl child; credentials stay inside the hook.
        result = subprocess.run(['timeout', '30s', 'bash', str(hook), message],
                                capture_output=True, text=True, encoding='utf-8')
        accepted = re.search(r'전송 성공 \((200|204)\)', result.stdout)
        receipt.update(status='sent' if result.returncode == 0 and accepted else 'failed',
                       returncode=result.returncode,
                       http_status=int(accepted[1]) if accepted else None)
        if result.returncode == 124:
            receipt['status'] = 'delivery_unknown'
    except OSError as exc:
        receipt.update(status='failed', error_type=type(exc).__name__)
    receipt_path.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + '\n')
    print(f"Discord notification: {receipt['status']}; receipt: {receipt_path}", flush=True)
    return receipt
