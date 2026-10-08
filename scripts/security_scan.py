"""Scan a final Docker image with pinned Trivy/Grype; fail on any high/critical.

Scanners receive an image archive and private cache directories, never the Docker
socket, model volumes, credentials or GPUs. This is a developer/CI tool only.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile

TRIVY = 'aquasec/trivy@sha256:af6acf9a6b85dfe389a1941505c0ce9efef52a4719635e1a962f022a3d855daa'
GRYPE = 'anchore/grype@sha256:5c88961f4130e830542d441c7ed6c78baa28e799163abac53d2be4923fb5ab7d'


def findings(scanner, report):
    if scanner == 'trivy':
        if not isinstance(report.get('Results'), list):
            raise ValueError('Trivy results are missing; refusing to report a clean scan')
        return [{'id': v['VulnerabilityID'], 'package': v['PkgName'],
                 'installed': v['InstalledVersion'], 'severity': v['Severity'].upper(),
                 'fixed': v.get('FixedVersion'), 'target': row['Target']}
                for row in report['Results'] for v in row.get('Vulnerabilities', [])]
    if not isinstance(report.get('matches'), list):
        raise ValueError('Grype matches are missing; refusing to report a clean scan')
    status = report.get('descriptor', {}).get('db', {}).get('status', {})
    if status.get('valid') is not True:
        raise ValueError('Grype database is invalid or missing')
    return [{'id': m['vulnerability']['id'], 'package': m['artifact']['name'],
             'installed': m['artifact']['version'], 'severity': m['vulnerability']['severity'].upper(),
             'fixed': m['vulnerability'].get('fix', {}).get('versions', []),
             'target': [v['path'] for v in m['artifact'].get('locations', [])]}
            for m in report['matches']]


def summarize(scanner, report):
    rows = findings(scanner, report)
    priority = [v for v in rows if v['severity'] in {'CRITICAL', 'HIGH'}]
    return {'counts': dict(Counter(v['severity'] for v in rows)),
            'priority_findings': priority,
            'priority_advisories': sorted({v['id'] for v in priority}),
            'passed': not priority}


def execute(command, output=None, log=None):
    if output is None:
        subprocess.run(command, check=True)
    else:
        with Path(output).open('w') as out, Path(log).open('w') as err:
            subprocess.run(command, stdout=out, stderr=err, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--image', help='Local Docker image reference or ID')
    source.add_argument('--archive', type=Path, help='Existing docker-save archive')
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--cache-dir', type=Path, help='Reuse an existing scanner cache')
    parser.add_argument('--skip-db-update', action='store_true', help='Use the recorded DB snapshot for before/after comparisons')
    args = parser.parse_args()
    if args.image and not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9./_:@-]*', args.image):
        parser.error('Invalid Docker image reference')
    out = args.output_dir.resolve(); out.mkdir(parents=True, exist_ok=True)
    cache = (args.cache_dir or (out / '.cache')).resolve()
    for scanner in ['trivy', 'grype']: (cache / scanner).mkdir(parents=True, exist_ok=True)
    for scanner in [TRIVY, GRYPE]:
        if subprocess.run(['docker', 'image', 'inspect', scanner], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode:
            execute(['docker', 'pull', scanner])
    metadata = {'started_utc': datetime.now(timezone.utc).isoformat(), 'scanner_images': {'trivy': TRIVY, 'grype': GRYPE}}
    with tempfile.TemporaryDirectory(prefix='clef-image-scan-', dir=out) as temporary:
        archive = args.archive.resolve() if args.archive else Path(temporary) / 'image.tar'
        if args.image:
            metadata['image_id'] = subprocess.check_output(['docker', 'image', 'inspect', '--format', '{{.Id}}', args.image], text=True).strip()
            execute(['docker', 'save', '--output', str(archive), args.image])
        if not archive.is_file(): raise ValueError('Image archive does not exist')
        metadata['archive_bytes'] = archive.stat().st_size
        reports = {}
        for scanner, image in [('trivy', TRIVY), ('grype', GRYPE)]:
            scratch = Path(temporary) / (scanner + '-tmp'); scratch.mkdir()
            base = ['docker', 'run', '--rm', '--cpus', '2', '--memory', '6g',
                    '--user', f'{os.getuid()}:{os.getgid()}', '--cap-drop', 'ALL',
                    '--security-opt', 'no-new-privileges=true', '-v', str(cache/scanner)+':/cache',
                    '-v', str(scratch)+':/tmp']
            if scanner == 'trivy':
                tool = [image, 'image', '--cache-dir', '/cache']
                update = tool + ['--download-db-only', '--no-progress']
                scan = tool + ['--skip-db-update', '--offline-scan', '--scanners', 'vuln',
                               '--parallel', '2', '--timeout', '20m', '--format', 'json', '--input', '/scan/image.tar']
            else:
                base += ['-e', 'GRYPE_DB_CACHE_DIR=/cache', '-e', 'GRYPE_DB_AUTO_UPDATE=false']
                update = [image, 'db', 'update']
                scan = [image, 'docker-archive:/scan/image.tar', '--scope', 'squashed', '-o', 'json']
            # Complete database initialization before scanning; no update races.
            if not args.skip_db_update:
                execute(base+update, out/(scanner+'-db.txt'), out/(scanner+'-db.log'))
            raw, log = out/(scanner+'.json'), out/(scanner+'.log')
            execute(base+['-v', str(archive)+':/scan/image.tar:ro']+scan, raw, log)
            report = json.loads(raw.read_text())
            reports[scanner] = summarize(scanner, report)
            reports[scanner]['database'] = report.get('descriptor', {}).get('db') if scanner == 'grype' else None
            if scanner == 'trivy':
                execute(base+[image, '--cache-dir', '/cache', '--version'], out/'trivy-version.txt', out/'trivy-version.log')
        result = {**metadata, 'reports': reports, 'passed': all(v['passed'] for v in reports.values())}
        (out/'summary.json').write_text(json.dumps(result, indent=2)+'\n')
        for scanner, report in reports.items(): print(scanner, report['counts'], 'PASS' if report['passed'] else 'FAIL')
        return 0 if result['passed'] else 1


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (OSError, ValueError, KeyError, subprocess.CalledProcessError) as exc:
        print('Security scan failed:', str(exc))
        raise SystemExit(2)
