"""Optional server-side Bearer authentication; no GPU dependencies."""
import hmac
import hashlib
import os
import secrets
import time
from http.cookies import SimpleCookie, CookieError
from pathlib import Path
from urllib.parse import urlsplit

from starlette.responses import JSONResponse, Response


def configured_api_key(environ=None):
    env = os.environ if environ is None else environ
    value, file = env.get('CLEF_API_KEY', ''), env.get('CLEF_API_KEY_FILE', '')
    if value and file:
        raise ValueError('Set only one of CLEF_API_KEY or CLEF_API_KEY_FILE')
    if file:
        try:
            with Path(file).open('rb') as stream:
                raw = stream.read(65537)
            if len(raw) > 65536:
                raise ValueError('API key file is too large')
            value = raw.decode('ascii').rstrip('\r\n')
        except (OSError, UnicodeError):
            raise ValueError('Cannot read an ASCII API key file') from None
        if not value:
            raise ValueError('API key file is empty')
    if value and any(ord(char) < 33 or ord(char) > 126 for char in value):
        raise ValueError('API key must contain printable ASCII without spaces')
    if env.get('CLEF_REQUIRE_API_KEY', '0') == '1' and not value:
        raise ValueError('API key required: set CLEF_API_KEY or CLEF_API_KEY_FILE')
    return value


class APIKeyMiddleware:
    # Public portal assets and documentation contain no key. Detailed runtime
    # health and every inference/discovery endpoint remain protected.
    PUBLIC_PATHS = {'/', '/ui-config', '/readyz', '/docs', '/docs/oauth2-redirect', '/redoc', '/openapi.json'}

    COOKIE_NAME = 'clef_portal_session'

    def __init__(self, app, api_key='', portal_auto_auth=True, cookie_secure=False, session_seconds=28800):
        self.app = app
        self.api_key = api_key.encode('ascii')
        self.portal_auto_auth = portal_auto_auth
        self.cookie_secure = cookie_secure
        self.session_seconds = session_seconds
        self.session_secret = secrets.token_bytes(32)

    def session_token(self):
        message = f'{int(time.time()) + self.session_seconds}.{secrets.token_hex(16)}'
        signature = hmac.new(self.session_secret, message.encode(), hashlib.sha256).hexdigest()
        return message + '.' + signature

    def portal_authorized(self, scope):
        headers = scope.get('headers', [])
        # Custom same-origin header prevents cookie-backed cross-site form
        # submissions; direct API clients still authenticate with Bearer.
        if [v for k,v in headers if k.lower()==b'x-clef-portal'] != [b'1']:
            return False
        origins = [v for k,v in headers if k.lower()==b'origin']
        hosts = [v for k,v in headers if k.lower()==b'host']
        if origins:
            if len(origins)!=1 or len(hosts)!=1:
                return False
            try:
                origin = urlsplit(origins[0].decode('ascii'))
                host = urlsplit(origin.scheme + '://' + hosts[0].decode('ascii'))
                port = 443 if origin.scheme == 'https' else 80
                if origin.scheme not in {'http','https'} or origin.username or origin.password or origin.path or origin.query or origin.fragment:
                    return False
                if (origin.hostname, origin.port or port) != (host.hostname, host.port or port):
                    return False
            except (UnicodeError, ValueError):
                return False
        try:
            raw = [v for k,v in headers if k.lower()==b'cookie']
            if len(raw)!=1 or len(raw[0])>4096:
                return False
            cookies = SimpleCookie(); cookies.load(raw[0].decode('ascii'))
            token = cookies[self.COOKIE_NAME].value
            expires, nonce, signature = token.split('.')
            expiry = int(expires)
            if expiry <= time.time() or expiry > time.time()+self.session_seconds+1 or len(nonce)!=32:
                return False
            expected = hmac.new(self.session_secret, (expires+'.'+nonce).encode(), hashlib.sha256).hexdigest()
            return hmac.compare_digest(signature, expected)
        except (KeyError, ValueError, UnicodeError, CookieError):
            return False

    async def __call__(self, scope, receive, send):
        if scope['type'] != 'http' or not self.api_key:
            return await self.app(scope, receive, send)
        path = scope.get('path', '')
        public = scope.get('method') in {'GET', 'HEAD'} and (path in self.PUBLIC_PATHS or path.startswith('/ui/'))
        if public:
            if path=='/' and scope.get('method')=='GET' and self.portal_auto_auth:
                cookie = Response()
                cookie.set_cookie(self.COOKIE_NAME, self.session_token(), max_age=self.session_seconds,
                                  httponly=True, secure=self.cookie_secure or scope.get('scheme')=='https', samesite='strict', path='/')
                async def portal_send(message):
                    if message['type']=='http.response.start' and message['status']==200:
                        message = {**message, 'headers': [(k,v) for k,v in message.get('headers',[]) if k.lower()!=b'cache-control']
                                   + [(b'cache-control',b'private, no-store')]
                                   + [(k,v) for k,v in cookie.raw_headers if k.lower()==b'set-cookie']}
                    await send(message)
                return await self.app(scope, receive, portal_send)
            return await self.app(scope, receive, send)
        headers = [v for k, v in scope.get('headers', []) if k.lower() == b'authorization']
        candidate = b''
        if len(headers) == 1:
            parts = headers[0].split(b' ', 1)
            if len(parts) == 2 and parts[0].lower() == b'bearer':
                candidate = parts[1]
        authorized = bool(candidate) and hmac.compare_digest(candidate, self.api_key)
        if not headers and self.portal_auto_auth:
            authorized = self.portal_authorized(scope)
        if not authorized:
            response = JSONResponse({'detail': 'A valid API key is required.'}, status_code=401,
                                    headers={'WWW-Authenticate': 'Bearer', 'Cache-Control': 'no-store'})
            return await response(scope, receive, send)
        return await self.app(scope, receive, send)
