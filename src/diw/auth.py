from __future__ import annotations

from dataclasses import dataclass
import importlib
import os
from typing import Protocol


@dataclass(frozen=True)
class AuthenticatedPrincipal:
    subject: str
    tenant_id: str | None
    claims: dict[str, object]


class TokenAuthenticator(Protocol):
    def authenticate(self, token: str) -> AuthenticatedPrincipal:
        ...


class OIDCAuthenticator:
    """Validate OIDC bearer tokens against issuer JWKS and required claims."""

    def __init__(
        self,
        *,
        issuer: str,
        audience: str,
        jwks_url: str,
        tenant_claim: str = "tenant_id",
    ) -> None:
        if not issuer or not audience or not jwks_url:
            raise ValueError("OIDC issuer, audience, and JWKS URL are required")
        if not tenant_claim:
            raise ValueError("OIDC tenant claim name is required")
        self.issuer = issuer.rstrip("/")
        self.audience = audience
        self.jwks_url = jwks_url
        self.tenant_claim = tenant_claim
        self._jwk_client = None

    @classmethod
    def from_env(cls) -> OIDCAuthenticator:
        return cls(
            issuer=os.getenv("OIDC_ISSUER", ""),
            audience=os.getenv("OIDC_AUDIENCE", ""),
            jwks_url=os.getenv("OIDC_JWKS_URL", ""),
            tenant_claim=os.getenv("OIDC_TENANT_CLAIM", "tenant_id"),
        )

    def authenticate(self, token: str) -> AuthenticatedPrincipal:
        try:
            jwt = importlib.import_module("jwt")
        except ImportError as error:
            raise RuntimeError(
                "OIDC support requires the 'auth' extra: pip install -e '.[auth]'"
            ) from error
        if self._jwk_client is None:
            self._jwk_client = jwt.PyJWKClient(self.jwks_url)
        signing_key = self._jwk_client.get_signing_key_from_jwt(token)
        claims = jwt.decode(
            token,
            signing_key.key,
            algorithms=["RS256", "ES256"],
            audience=self.audience,
            issuer=self.issuer,
            options={"require": ["exp", "iss", "aud", "sub"]},
        )
        subject = claims.get("sub")
        tenant_id = claims.get(self.tenant_claim)
        if not isinstance(subject, str) or not subject:
            raise ValueError("OIDC token subject is missing")
        if not isinstance(tenant_id, str) or not tenant_id:
            raise ValueError(f"OIDC token tenant claim '{self.tenant_claim}' is missing")
        return AuthenticatedPrincipal(
            subject=subject,
            tenant_id=tenant_id,
            claims=dict(claims),
        )


class GoogleOIDCAuthenticator:
    """Verify Google OAuth ID tokens issued for one web client."""

    def __init__(self, *, audience: str) -> None:
        if not audience:
            raise ValueError("GOOGLE_OAUTH_CLIENT_ID is required")
        self.audience = audience

    @classmethod
    def from_env(cls) -> GoogleOIDCAuthenticator:
        return cls(audience=os.getenv("GOOGLE_OAUTH_CLIENT_ID", ""))

    def authenticate(self, token: str) -> AuthenticatedPrincipal:
        try:
            google_id_token = importlib.import_module("google.oauth2.id_token")
            google_requests = importlib.import_module("google.auth.transport.requests")
        except ImportError as error:
            raise RuntimeError(
                "Google OIDC support requires the 'cloud' extra: pip install -e '.[cloud]'"
            ) from error
        claims = google_id_token.verify_oauth2_token(
            token,
            google_requests.Request(),
            self.audience,
        )
        subject = claims.get("sub")
        if not isinstance(subject, str) or not subject:
            raise ValueError("Google ID token subject is missing")
        firebase = claims.get("firebase")
        firebase_tenant = firebase.get("tenant") if isinstance(firebase, dict) else None
        tenant_id = claims.get("tenant_id") or firebase_tenant
        if tenant_id is not None and not isinstance(tenant_id, str):
            raise ValueError("Google ID token tenant claim is invalid")
        return AuthenticatedPrincipal(
            subject=subject,
            tenant_id=tenant_id,
            claims=dict(claims),
        )
