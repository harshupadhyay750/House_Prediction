import pytest

from src.supabase_store import SupabaseError, SupabaseRESTClient


class FakeResponse:
    ok = True
    status_code = 200

    def __init__(self, payload):
        self.payload = payload

    def json(self):
        return self.payload


def test_sign_in_uses_supabase_password_grant(monkeypatch):
    captured = {}

    def fake_request(method, url, **kwargs):
        captured.update(method=method, url=url, **kwargs)
        return FakeResponse({"access_token": "session-token"})

    monkeypatch.setattr("src.supabase_store.requests.request", fake_request)
    client = SupabaseRESTClient("https://example.supabase.co", "anon-key")

    result = client.sign_in("user@example.com", "a-long-password")

    assert result["access_token"] == "session-token"
    assert captured["url"].endswith("/auth/v1/token")
    assert captured["params"] == {"grant_type": "password"}
    assert captured["json"] == {"email": "user@example.com", "password": "a-long-password"}
    assert captured["headers"]["apikey"] == "anon-key"


def test_sign_up_uses_configured_email_callback(monkeypatch):
    captured = {}

    def fake_request(method, url, **kwargs):
        captured.update(method=method, url=url, **kwargs)
        return FakeResponse({"user": {"id": "user-id"}})

    monkeypatch.setattr("src.supabase_store.requests.request", fake_request)
    client = SupabaseRESTClient("https://example.supabase.co", "anon-key")

    client.sign_up("user@example.com", "a-long-password", "Prop IQ", "https://propiq.example")

    assert captured["url"].endswith("/auth/v1/signup")
    assert captured["params"] == {"redirect_to": "https://propiq.example"}
    assert captured["json"]["data"] == {"display_name": "Prop IQ"}


def test_saved_valuation_binds_owner_and_does_not_accept_email(monkeypatch):
    captured = {}

    def fake_request(method, url, **kwargs):
        captured.update(method=method, url=url, **kwargs)
        return FakeResponse([])

    monkeypatch.setattr("src.supabase_store.requests.request", fake_request)
    client = SupabaseRESTClient("https://example.supabase.co", "anon-key")
    payload = {"Area": 1800, "Bedrooms": 3}
    result = {"predicted_price": 420000.0, "price_formatted": "$420,000", "currency": "USD"}

    client.save_valuation("user-token", "user-id", payload, result)

    assert captured["headers"]["Authorization"] == "Bearer user-token"
    assert captured["json"]["user_id"] == "user-id"
    assert captured["json"]["property_data"] == payload
    assert "user_email" not in captured["json"]


def test_non_admin_cannot_reach_admin_user_endpoint(monkeypatch):
    client = SupabaseRESTClient("https://example.supabase.co", "anon-key", "service-key")
    client.get_user = lambda _token: {"id": "user-id"}
    requested_paths = []

    def fake_internal_request(method, path, **kwargs):
        requested_paths.append(path)
        return [{"role": "user"}]

    client._request = fake_internal_request

    with pytest.raises(SupabaseError, match="Administrator access"):
        client.list_admin_users("user-token")

    assert "/auth/v1/admin/users" not in requested_paths


def test_recovery_token_is_verified_by_auth_server(monkeypatch):
    captured = {}

    def fake_request(method, url, **kwargs):
        captured.update(method=method, url=url, **kwargs)
        return FakeResponse({"access_token": "recovery-session"})

    monkeypatch.setattr("src.supabase_store.requests.request", fake_request)
    client = SupabaseRESTClient("https://example.supabase.co", "anon-key")

    result = client.verify_email_token("one-time-hash", "recovery")

    assert result["access_token"] == "recovery-session"
    assert captured["url"].endswith("/auth/v1/verify")
    assert captured["json"] == {"token_hash": "one-time-hash", "type": "recovery"}


def test_password_update_requires_the_user_session(monkeypatch):
    captured = {}

    def fake_request(method, url, **kwargs):
        captured.update(method=method, url=url, **kwargs)
        return FakeResponse({})

    monkeypatch.setattr("src.supabase_store.requests.request", fake_request)
    client = SupabaseRESTClient("https://example.supabase.co", "anon-key")

    client.update_password("user-session", "a-new-long-password")

    assert captured["method"] == "PUT"
    assert captured["url"].endswith("/auth/v1/user")
    assert captured["headers"]["Authorization"] == "Bearer user-session"
    assert captured["json"] == {"password": "a-new-long-password"}