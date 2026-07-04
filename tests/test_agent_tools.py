import httpx
import pytest

from demandcast.agent.tools import build_forecast_tool


class _FakeSuccessTransport(httpx.BaseTransport):
    def handle_request(self, request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "store_id": 262,
                "date": "2015-08-01",
                "predicted_sales": 19752.06,
                "model_name": "demandcast-lightgbm",
                "model_version": "1",
            },
        )


class _FakeNotFoundTransport(httpx.BaseTransport):
    def handle_request(self, request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, json={"detail": "No sales history found for store_id=999"})


class _FakeUnreachableTransport(httpx.BaseTransport):
    def handle_request(self, request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)


@pytest.fixture(autouse=True)
def _patch_httpx_post(monkeypatch):
    """The tool calls the bare `httpx.post` module function; patch it to route
    through a fake transport per-test instead of hitting a real server."""

    def make_patched_post(transport_cls):
        def patched_post(url, json=None, timeout=None):
            with httpx.Client(transport=transport_cls()) as client:
                return client.post(url, json=json, timeout=timeout)

        return patched_post

    return make_patched_post


def test_forecast_tool_returns_prediction_on_success(monkeypatch, _patch_httpx_post):
    monkeypatch.setattr(httpx, "post", _patch_httpx_post(_FakeSuccessTransport))
    tool = build_forecast_tool()

    result = tool.invoke({"store_id": 262, "date": "2015-08-01", "is_promo": True})

    assert "19,752" in result
    assert "store 262" in result
    assert "demandcast-lightgbm" in result


def test_forecast_tool_reports_api_error_without_raising(monkeypatch, _patch_httpx_post):
    monkeypatch.setattr(httpx, "post", _patch_httpx_post(_FakeNotFoundTransport))
    tool = build_forecast_tool()

    result = tool.invoke({"store_id": 999, "date": "2015-08-01"})

    assert "error" in result.lower()
    assert "999" in result


def test_forecast_tool_reports_unreachable_api_without_raising(monkeypatch, _patch_httpx_post):
    monkeypatch.setattr(httpx, "post", _patch_httpx_post(_FakeUnreachableTransport))
    tool = build_forecast_tool()

    result = tool.invoke({"store_id": 262, "date": "2015-08-01"})

    assert "could not reach" in result.lower()


def test_forecast_tool_rejects_non_positive_store_id():
    tool = build_forecast_tool()
    with pytest.raises(Exception):  # noqa: B017 - pydantic ValidationError, wrapped by the tool
        tool.invoke({"store_id": 0, "date": "2015-08-01"})
