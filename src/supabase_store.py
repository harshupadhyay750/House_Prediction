"""Server-side Supabase Auth and PostgREST access for the Streamlit UI."""

from typing import Any, Dict, List, Optional

import requests


class SupabaseError(RuntimeError):
    """A safe, user-displayable Supabase request error."""


class SupabaseRESTClient:
    def __init__(self, project_url: str, anon_key: str, service_role_key: Optional[str] = None):
        self.project_url = project_url.rstrip("/")
        self.anon_key = anon_key
        self.service_role_key = service_role_key

    def _request(
        self,
        method: str,
        path: str,
        *,
        access_token: Optional[str] = None,
        body: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, Any]] = None,
        prefer: Optional[str] = None,
        use_service_role: bool = False,
    ) -> Any:
        if use_service_role and not self.service_role_key:
            raise SupabaseError("Admin features are not configured on this server.")

        api_key = self.service_role_key if use_service_role else self.anon_key
        bearer = api_key if use_service_role else (access_token or self.anon_key)
        headers = {
            "apikey": api_key,
            "Authorization": f"Bearer {bearer}",
            "Content-Type": "application/json",
        }
        if prefer:
            headers["Prefer"] = prefer

        try:
            response = requests.request(
                method,
                f"{self.project_url}{path}",
                headers=headers,
                json=body,
                params=params,
                timeout=15,
            )
        except requests.RequestException as exc:
            raise SupabaseError("Could not connect to the account service. Please try again.") from exc

        try:
            data = response.json()
        except ValueError:
            data = None

        if not response.ok:
            if isinstance(data, dict):
                message = data.get("msg") or data.get("message") or data.get("error_description") or data.get("error")
            else:
                message = None
            raise SupabaseError(message or f"Account service request failed ({response.status_code}).")
        return data

    def sign_up(
        self,
        email: str,
        password: str,
        display_name: str,
        redirect_to: Optional[str] = None,
    ) -> Dict[str, Any]:
        return self._request(
            "POST",
            "/auth/v1/signup",
            params={"redirect_to": redirect_to} if redirect_to else None,
            body={"email": email, "password": password, "data": {"display_name": display_name}},
        )

    def sign_in(self, email: str, password: str) -> Dict[str, Any]:
        return self._request(
            "POST",
            "/auth/v1/token",
            params={"grant_type": "password"},
            body={"email": email, "password": password},
        )

    def refresh_session(self, refresh_token: str) -> Dict[str, Any]:
        return self._request(
            "POST",
            "/auth/v1/token",
            params={"grant_type": "refresh_token"},
            body={"refresh_token": refresh_token},
        )

    def request_password_reset(self, email: str, redirect_to: Optional[str] = None) -> None:
        params = {"redirect_to": redirect_to} if redirect_to else None
        self._request("POST", "/auth/v1/recover", params=params, body={"email": email})

    def verify_email_token(self, token_hash: str, token_type: str) -> Dict[str, Any]:
        if token_type not in {"email", "recovery"}:
            raise SupabaseError("This email link type is not supported.")
        return self._request(
            "POST",
            "/auth/v1/verify",
            body={"token_hash": token_hash, "type": token_type},
        )

    def update_password(self, access_token: str, password: str) -> None:
        self._request(
            "PUT",
            "/auth/v1/user",
            access_token=access_token,
            body={"password": password},
        )

    def sign_out(self, access_token: str) -> None:
        self._request("POST", "/auth/v1/logout", access_token=access_token)

    def get_user(self, access_token: str) -> Dict[str, Any]:
        return self._request("GET", "/auth/v1/user", access_token=access_token)

    def get_profile(self, access_token: str, user_id: str) -> Optional[Dict[str, Any]]:
        rows = self._request(
            "GET",
            "/rest/v1/profiles",
            access_token=access_token,
            params={"select": "id,display_name,role", "id": f"eq.{user_id}", "limit": 1},
        )
        return rows[0] if rows else None

    def update_profile(self, access_token: str, user_id: str, display_name: str) -> None:
        self._request(
            "PATCH",
            "/rest/v1/profiles",
            access_token=access_token,
            params={"id": f"eq.{user_id}"},
            body={"display_name": display_name.strip()},
            prefer="return=minimal",
        )

    def save_valuation(
        self,
        access_token: str,
        user_id: str,
        property_data: Dict[str, Any],
        result: Dict[str, Any],
    ) -> None:
        self._request(
            "POST",
            "/rest/v1/valuations",
            access_token=access_token,
            body={
                "user_id": user_id,
                "currency": result.get("currency", property_data.get("Currency", "USD")),
                "country": result.get("country", property_data.get("Country")),
                "city": result.get("city", property_data.get("City")),
                "predicted_price": result["predicted_price"],
                "price_formatted": result["price_formatted"],
                "property_data": property_data,
                "result_data": result,
            },
            prefer="return=minimal",
        )

    def list_valuations(self, access_token: str) -> List[Dict[str, Any]]:
        return self._request(
            "GET",
            "/rest/v1/valuations",
            access_token=access_token,
            params={
                "select": "id,created_at,currency,country,city,predicted_price,price_formatted,property_data,result_data",
                "order": "created_at.desc",
                "limit": 100,
            },
        )

    def delete_valuation(self, access_token: str, valuation_id: str) -> None:
        self._request(
            "DELETE",
            "/rest/v1/valuations",
            access_token=access_token,
            params={"id": f"eq.{valuation_id}"},
            prefer="return=minimal",
        )

    def get_market_settings(self) -> Optional[Dict[str, Any]]:
        rows = self._request(
            "GET",
            "/rest/v1/market_settings",
            params={"select": "enabled_countries,enabled_currencies", "is_singleton": "eq.true", "limit": 1},
        )
        return rows[0] if rows else None

    def _require_admin(self, access_token: str) -> str:
        user = self.get_user(access_token)
        user_id = user.get("id")
        if not user_id:
            raise SupabaseError("Your session is invalid. Please sign in again.")
        rows = self._request(
            "GET",
            "/rest/v1/profiles",
            params={"select": "role", "id": f"eq.{user_id}", "limit": 1},
            use_service_role=True,
        )
        if not rows or rows[0].get("role") != "admin":
            raise SupabaseError("Administrator access is required for this action.")
        return user_id

    def verify_admin(self, access_token: str) -> str:
        return self._require_admin(access_token)

    def list_admin_users(self, access_token: str, page: int = 1) -> List[Dict[str, Any]]:
        self._require_admin(access_token)
        result = self._request(
            "GET",
            "/auth/v1/admin/users",
            params={"page": page, "per_page": 100},
            use_service_role=True,
        )
        return result.get("users", []) if isinstance(result, dict) else []

    def set_user_banned(self, access_token: str, user_id: str, banned: bool) -> None:
        self._require_admin(access_token)
        self._request(
            "PUT",
            f"/auth/v1/admin/users/{user_id}",
            body={"ban_duration": "876000h" if banned else "none"},
            use_service_role=True,
        )

    def list_admin_valuations(self, access_token: str) -> List[Dict[str, Any]]:
        self._require_admin(access_token)
        return self._request(
            "GET",
            "/rest/v1/valuations",
            params={
                "select": "id,user_id,created_at,currency,country,city,predicted_price,price_formatted,profiles(email,display_name)",
                "order": "created_at.desc",
                "limit": 200,
            },
            use_service_role=True,
        )

    def save_market_settings(
        self,
        access_token: str,
        enabled_countries: List[str],
        enabled_currencies: List[str],
    ) -> None:
        admin_id = self._require_admin(access_token)
        if not enabled_countries or not enabled_currencies:
            raise SupabaseError("Keep at least one country and currency enabled.")
        self._request(
            "POST",
            "/rest/v1/market_settings",
            params={"on_conflict": "is_singleton"},
            body={
                "is_singleton": True,
                "enabled_countries": enabled_countries,
                "enabled_currencies": enabled_currencies,
                "updated_by": admin_id,
            },
            prefer="resolution=merge-duplicates,return=minimal",
            use_service_role=True,
        )