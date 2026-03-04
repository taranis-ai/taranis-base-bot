import pytest
import respx


@pytest.mark.asyncio
async def test_health_check(client):
    response = await client.get("/health")
    assert response.status_code == 200
    data = await response.get_json()
    assert data == {"status": "ok"}


@pytest.mark.asyncio
async def test_modelinfo(client, custom_settings):
    with respx.mock(base_url="https://huggingface.co") as mock:
        mock.get(f"/api/models/{custom_settings.MODEL}").respond(
            status_code=200,
            json={"model": custom_settings.MODEL},
        )

        response = await client.get("/modelinfo")
        assert response.status_code == 200
        data = await response.get_json()
        assert data == {"model": custom_settings.MODEL}


@pytest.mark.asyncio
async def test_root_avail(client):
    response = await client.post("/", json={"text": "Health check text."})
    assert response.status_code == 200
    data = await response.get_json()
    assert isinstance(data, dict)


@pytest.mark.asyncio
async def test_request_parser_success(client_with_request_parser):
    response = await client_with_request_parser.post("/", json={"text": "A valid dict"})
    assert response.status_code == 200
    data = await response.get_json()
    assert isinstance(data, dict)
    assert data["text"] == "A valid dict"


@pytest.mark.asyncio
async def test_request_parser_invalid_input(client_with_request_parser):
    response = await client_with_request_parser.post("/", json={"another_key": "An invalid dict"})
    assert response.status_code == 400
    data = await response.get_json()
    assert isinstance(data, dict)
    assert data["error"] == "Could not parse payload. Check bot logs for more details."


@pytest.mark.asyncio
async def test_predict_success(client_with_predict):
    response = await client_with_predict.post("/", json={"text": "a" * 15})
    assert response.status_code == 200
    result = await response.get_json()
    assert isinstance(result, dict)
    assert result["len"] == 15


@pytest.mark.asyncio
async def test_predict_invalid_input(client_with_predict):
    response = await client_with_predict.post("/", json={"text": ("this", "is", "not", "a", "string")})
    result = await response.get_json()
    assert isinstance(result, dict)
    assert result["error"] == "Bot execution failed. Check bot logs for more details."


@pytest.mark.asyncio
async def test_invalid_json(client):
    response = await client.post("/", data="notjson", headers={"Content-Type": "application/json"})
    assert response.status_code == 400


@pytest.mark.asyncio
async def test_valid_json_but_not_dict(client):
    """Test valid JSON that isn't a dictionary"""
    # This mimics: curl -d '"notjson"' (valid JSON string, not a dict)
    response = await client.post("/", data='"notjson"', headers={"Content-Type": "application/json"})
    assert response.status_code == 400
    data = await response.get_json()
    assert data["error"] == "Payload must be a dict!"


@pytest.mark.asyncio
async def test_missing_authorization_header_with_api_key(client_with_api_key):
    """Test missing Authorization header when API key is required"""
    response = await client_with_api_key.post("/", json={"text": "test"})
    assert response.status_code == 401
    data = await response.get_json()
    assert data["error"] == "not authorized"


@pytest.mark.asyncio
async def test_malformed_authorization_header_syntax(client_with_api_key):
    """Test malformed Authorization header - the exact issue you encountered"""
    # This mimics: curl -H "Authorization Bearer: test-api-key-123" (missing colon after Authorization)
    headers = {"Authorization Bearer": "test-api-key-123"}
    response = await client_with_api_key.post("/", json={"text": "test"}, headers=headers)
    assert response.status_code == 401
    data = await response.get_json()
    assert data["error"] == "not authorized"


@pytest.mark.asyncio
async def test_malformed_authorization_header_no_bearer(client_with_api_key):
    """Test Authorization header without Bearer prefix"""
    headers = {"Authorization": "test-api-key-123"}  # Missing "Bearer " prefix
    response = await client_with_api_key.post("/", json={"text": "test"}, headers=headers)
    assert response.status_code == 401
    data = await response.get_json()
    assert data["error"] == "not authorized"


@pytest.mark.asyncio
async def test_correct_authorization_header(client_with_api_key):
    """Test correct Authorization header format"""
    headers = {"Authorization": "Bearer test-api-key-123"}
    response = await client_with_api_key.post("/", json={"text": "test"}, headers=headers)
    assert response.status_code == 200
    data = await response.get_json()
    assert isinstance(data, dict)


@pytest.mark.asyncio
async def test_wrong_api_key(client_with_api_key):
    """Test wrong API key"""
    headers = {"Authorization": "Bearer wrong-api-key"}
    response = await client_with_api_key.post("/", json={"text": "test"}, headers=headers)
    assert response.status_code == 401
    data = await response.get_json()
    assert data["error"] == "not authorized"


@pytest.mark.asyncio
async def test_modelinfo_func_ok(client_with_modelinfo_fn, custom_settings):
    with respx.mock(base_url="https://huggingface.co") as mock:
        mock.get(f"/api/models/{custom_settings.MODEL}").respond(
            status_code=200,
            json={"model": custom_settings.MODEL, "ok": True},
        )

        response = await client_with_modelinfo_fn.get("/modelinfo")
        assert response.status_code == 200
        data = await response.get_json()
        assert data == {"model": custom_settings.MODEL, "ok": True}


@pytest.mark.asyncio
async def test_modelinfo_func_error(client_with_modelinfo_fn, custom_settings):
    with respx.mock(base_url="https://huggingface.co") as mock:
        mock.get(f"/api/models/{custom_settings.MODEL}").mock(side_effect=Exception("Connection failed"))

        response = await client_with_modelinfo_fn.get("/modelinfo")
        assert response.status_code == 200
        data = await response.get_json()
        assert data == {
            "model": custom_settings.MODEL,
            "error": "Connection failed",
        }
