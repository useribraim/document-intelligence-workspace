import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from diw.auth import GoogleOIDCAuthenticator, OIDCAuthenticator


class OIDCAuthenticatorTests(unittest.TestCase):
    def test_validates_signature_and_required_oidc_claims(self):
        recorded: dict = {}

        class FakeJWKClient:
            def __init__(self, url: str):
                recorded["jwks_url"] = url

            def get_signing_key_from_jwt(self, token: str):
                recorded["token"] = token
                return SimpleNamespace(key="public-key")

        def decode(token, key, **kwargs):
            recorded["decode"] = {"token": token, "key": key, **kwargs}
            return {
                "sub": "user-123",
                "tenant_id": "tenant-456",
                "iss": "https://issuer.example",
                "aud": "diw",
                "exp": 9_999_999_999,
            }

        fake_jwt = SimpleNamespace(PyJWKClient=FakeJWKClient, decode=decode)
        with patch.dict(sys.modules, {"jwt": fake_jwt}):
            authenticator = OIDCAuthenticator(
                issuer="https://issuer.example/",
                audience="diw",
                jwks_url="https://issuer.example/jwks",
            )
            principal = authenticator.authenticate("signed-token")

        self.assertEqual(principal.subject, "user-123")
        self.assertEqual(principal.tenant_id, "tenant-456")
        self.assertEqual(recorded["decode"]["audience"], "diw")
        self.assertEqual(recorded["decode"]["issuer"], "https://issuer.example")
        self.assertIn("exp", recorded["decode"]["options"]["require"])

    def test_missing_tenant_claim_is_rejected(self):
        fake_jwt = SimpleNamespace(
            PyJWKClient=lambda _: SimpleNamespace(
                get_signing_key_from_jwt=lambda __: SimpleNamespace(key="key")
            ),
            decode=lambda *args, **kwargs: {"sub": "user"},
        )
        with patch.dict(sys.modules, {"jwt": fake_jwt}):
            authenticator = OIDCAuthenticator(
                issuer="https://issuer.example",
                audience="diw",
                jwks_url="https://issuer.example/jwks",
            )
            with self.assertRaisesRegex(ValueError, "tenant claim"):
                authenticator.authenticate("token")


class GoogleOIDCAuthenticatorTests(unittest.TestCase):
    def test_verifies_google_id_token_for_configured_audience(self):
        recorded: dict = {}

        def verify(token, request, audience):
            recorded.update(token=token, request=request, audience=audience)
            return {
                "sub": "google-subject",
                "email": "person@example.com",
                "email_verified": True,
            }

        fake_id_token = SimpleNamespace(verify_oauth2_token=verify)
        fake_requests = SimpleNamespace(Request=lambda: "google-request")
        with patch.dict(
            sys.modules,
            {
                "google.oauth2.id_token": fake_id_token,
                "google.auth.transport.requests": fake_requests,
            },
        ):
            principal = GoogleOIDCAuthenticator(
                audience="client.apps.googleusercontent.com"
            ).authenticate("google-token")

        self.assertEqual(principal.subject, "google-subject")
        self.assertIsNone(principal.tenant_id)
        self.assertEqual(recorded["audience"], "client.apps.googleusercontent.com")


if __name__ == "__main__":
    unittest.main()
