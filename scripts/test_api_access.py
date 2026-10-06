"""Exercise authentication at the ASGI boundary before any request body/queue work."""
import asyncio
from pathlib import Path
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from clef_service.access import APIKeyMiddleware, configured_api_key


assert configured_api_key({}) == ''
assert configured_api_key({'CLEF_API_KEY': 'manually-chosen-key'}) == 'manually-chosen-key'
for env in [{'CLEF_REQUIRE_API_KEY': '1'}, {'CLEF_API_KEY': ' '},
            {'CLEF_API_KEY': 'bad\nkey'}, {'CLEF_API_KEY': 'non-ascii-\u00e9'},
            {'CLEF_API_KEY': 'one', 'CLEF_API_KEY_FILE': 'unused'}]:
    try:
        configured_api_key(env)
        raise AssertionError('Invalid credential configuration accepted')
    except ValueError:
        pass
with tempfile.TemporaryDirectory() as temp:
    key_file = Path(temp)/'secret'
    key_file.write_text('manual-file-key\n')
    assert configured_api_key({'CLEF_API_KEY_FILE': str(key_file), 'CLEF_REQUIRE_API_KEY': '1'}) == 'manual-file-key'
    for data in [b'', b'\n', b'bad key', b'\xff', b'x'*65537]:
        key_file.write_bytes(data)
        try:
            configured_api_key({'CLEF_API_KEY_FILE': str(key_file)})
            raise AssertionError('Invalid key file accepted')
        except ValueError:
            pass
    key_file.unlink()
    try:
        configured_api_key({'CLEF_API_KEY_FILE': str(key_file)})
        raise AssertionError('Missing key file accepted')
    except ValueError:
        pass


async def main():
    calls = []
    async def downstream(scope, receive, send):
        calls.append(scope['path'])
        await receive()
        await send({'type': 'http.response.start', 'status': 200, 'headers': []})
        await send({'type': 'http.response.body', 'body': b'ok'})
    app = APIKeyMiddleware(downstream, 'manual-test-key')
    async def request(path, headers=(), method='POST', enabled=app):
        messages, reads = [], []
        async def receive():
            reads.append(1)
            return {'type': 'http.request', 'body': b'oversized-untrusted-payload', 'more_body': False}
        async def send(message): messages.append(message)
        await enabled({'type': 'http', 'path': path, 'method': method, 'headers': list(headers)}, receive, send)
        return messages[0]['status'], reads, messages
    for path in ['/v1/systemone', '/v1/models', '/health', '/unknown', '/ui-config']:
        for headers in [(), ((b'authorization', b'Bearer wrong'),),
                        ((b'authorization', b'Basic manual-test-key'),),
                        ((b'authorization', b'Bearer manual-test-key '),),
                        ((b'authorization', b'Bearer manual-test-key'), (b'authorization', b'Bearer manual-test-key'))]:
            status, reads, messages = await request(path, headers)
            assert status == 401 and not reads
            assert (b'WWW-Authenticate', b'Bearer') in messages[0]['headers'] or (b'www-authenticate', b'Bearer') in messages[0]['headers']
    assert not calls
    for path in ['/v1/systemone', '/v1/models', '/health']:
        status, reads, _ = await request(path, ((b'authorization', b'bearer manual-test-key'),))
        assert status == 200 and reads
    for path in ['/', '/ui/app.js', '/ui-config', '/readyz', '/docs', '/openapi.json']:
        assert (await request(path, method='GET'))[0] == 200
    assert (await request('/v1/systemone?api_key=manual-test-key'))[0] == 401
    assert (await request('/v1/systemone', enabled=APIKeyMiddleware(downstream)))[0] == 200
    print('PASS: manually configured env/file keys, fail-closed validation, Bearer auth, public portal/docs/probe, body admission bypass, and auth-off compatibility')


asyncio.run(main())

# Swagger advertises Bearer security; middleware still enforces discovery and
# inference while its public shell/OpenAPI remain usable without a key.
from fastapi import FastAPI, Security
from fastapi.security import HTTPBearer
from fastapi.testclient import TestClient
fapp = FastAPI(dependencies=[Security(HTTPBearer(auto_error=False))])
fapp.add_middleware(APIKeyMiddleware, api_key="manual-test-key")
@fapp.get("/v1/models")
def models(): return {"data": [{"id": "clef"}]}
with TestClient(fapp) as client:
    assert client.get("/v1/models").status_code == 401
    assert client.get("/v1/models", headers={"Authorization": "Bearer manual-test-key"}).status_code == 200
    assert client.get("/docs").status_code == 200
    schema = client.get("/openapi.json").json()
    assert schema["components"]["securitySchemes"]["HTTPBearer"] == {"type": "http", "scheme": "bearer"}
    assert schema["paths"]["/v1/models"]["get"]["security"] == [{"HTTPBearer": []}]
print("PASS: HTTP discovery enforcement and Swagger Bearer authorization metadata")

# Automatic portal access uses an opaque session, never the configured API key.
from http.cookies import SimpleCookie
import hashlib,hmac,time
portal=FastAPI()
portal.add_middleware(APIKeyMiddleware, api_key='manual-test-key')
@portal.get('/')
def home(): return {'portal':True}
@portal.post('/v1/systemone')
def infer(body:dict): return {'accepted':body}
with TestClient(portal) as client:
    response=client.get('/')
    cookie=response.headers['set-cookie']
    assert 'HttpOnly' in cookie and 'SameSite=strict' in cookie and 'Max-Age=28800' in cookie
    assert 'manual-test-key' not in cookie+response.text
    assert response.headers['cache-control']=='private, no-store'
    assert client.post('/v1/systemone',json={}).status_code==401
    assert client.post('/v1/systemone',json={},headers={'X-Clef-Portal':'1'}).status_code==200
    assert client.post('/v1/systemone',json={},headers={'X-Clef-Portal':'1','Origin':'http://testserver'}).status_code==200
    assert client.post('/v1/systemone',json={},headers={'X-Clef-Portal':'1','Origin':'https://evil.example'}).status_code==401
    assert client.post('/v1/systemone',json={},headers={'X-Clef-Portal':'1','Authorization':'Bearer wrong'}).status_code==401
    token=client.cookies.get(APIKeyMiddleware.COOKIE_NAME)
    client.cookies.clear();client.cookies.set(APIKeyMiddleware.COOKIE_NAME,token[:-1]+('0' if token[-1]!='0' else '1'))
    assert client.post('/v1/systemone',json={},headers={'X-Clef-Portal':'1'}).status_code==401
with TestClient(portal,base_url='https://testserver') as client:
    assert 'Secure' in client.get('/').headers['set-cookie']
assert 'id="api-key-' not in (Path(__file__).resolve().parents[1]/'web/index.html').read_text()
assert "localStorage.setItem('clef.apiKey'" not in (Path(__file__).resolve().parents[1]/'web/app.js').read_text()

middleware=APIKeyMiddleware(lambda *args:None,api_key='manual-test-key')
scope={'headers':[(b'x-clef-portal',b'1')]}
token=middleware.session_token()
scope['headers'].append((b'cookie',(middleware.COOKIE_NAME+'='+token).encode()))
assert middleware.portal_authorized(scope)
assert not APIKeyMiddleware(lambda *args:None,api_key='manual-test-key').portal_authorized(scope), 'Restart must invalidate a session'
expired=str(int(time.time())-1)+'.'+'0'*32
signature=hmac.new(middleware.session_secret,expired.encode(),hashlib.sha256).hexdigest()
scope['headers'][-1]=(b'cookie',(middleware.COOKIE_NAME+'='+expired+'.'+signature).encode())
assert not middleware.portal_authorized(scope)
print('PASS: automatic HttpOnly portal session, no key editor/disclosure, CSRF checks, tamper/expiry/restart rejection, and HTTPS Secure cookie')
