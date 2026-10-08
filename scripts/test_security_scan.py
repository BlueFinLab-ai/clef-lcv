"""Scan gates must retain unfixed CVEs and fail closed on incomplete results."""
from security_scan import summarize, findings

t={'Results':[{'Target':'os','Vulnerabilities':[{'VulnerabilityID':'CVE-high','PkgName':'lib','InstalledVersion':'1','Severity':'HIGH'},
    {'VulnerabilityID':'CVE-medium','PkgName':'lib','InstalledVersion':'1','Severity':'MEDIUM','FixedVersion':'2'}]}]}
r=summarize('trivy',t)
assert not r['passed'] and r['counts']=={'HIGH':1,'MEDIUM':1}
assert r['priority_findings'][0]['fixed'] is None,'Unfixed high was hidden'
g={'matches':[{'vulnerability':{'id':'CVE-critical','severity':'Critical','fix':{'versions':[],'state':'not-fixed'}},
               'artifact':{'name':'python','version':'1','locations':[{'path':'/python'}]}}],
   'descriptor':{'db':{'status':{'valid':True}}}}
r=summarize('grype',g)
assert not r['passed'] and r['counts']=={'CRITICAL':1}
assert r['priority_advisories']==['CVE-critical']
for scanner,report in [('trivy',{}),('grype',{}),('grype',{'matches':[],'descriptor':{'db':{'status':{'valid':False}}}})]:
    try:findings(scanner,report)
    except ValueError:pass
    else:raise AssertionError('Missing scanner/DB results passed')
assert summarize('trivy',{'Results':[]})['passed']
assert summarize('grype',{'matches':[],'descriptor':{'db':{'status':{'valid':True}}}})['passed']
print('PASS: critical/high gates, unfixed findings, lower severity reporting and failed-scanner/DB rejection')
