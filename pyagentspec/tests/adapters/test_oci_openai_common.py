# Copyright © 2026 Oracle and/or its affiliates.
#
# This software is under the Apache License 2.0
# (LICENSE-APACHE or http://www.apache.org/licenses/LICENSE-2.0) or Universal Permissive License
# (UPL) 1.0 (LICENSE-UPL or https://oss.oracle.com/licenses/upl), at your option.
"""OCI Generative AI through the OpenAI-compatible API: signed ``openai`` clients, no network."""

import json
import sys
from pathlib import Path
from typing import Any, Dict

import httpx
import pytest

from pyagentspec.llms.ociclientconfig import (
    OciClientConfigWithApiKey,
    OciClientConfigWithInstancePrincipal,
    OciClientConfigWithResourcePrincipal,
    OciClientConfigWithSecurityToken,
)
from pyagentspec.llms.ocigenaiconfig import (
    ModelProvider,
    OciAPIType,
    OciGenAiConfig,
    ServingMode,
)
from pyagentspec.retrypolicy import RetryPolicy

pytest.importorskip("oci_openai")

from pyagentspec.adapters._oci_openai_common import (  # noqa: E402
    COMPARTMENT_ID_HEADER,
    CONVERSATION_STORE_ID_HEADER,
    OCI_OPENAI_ACCEPT_ENCODING,
    OCI_OPENAI_PLACEHOLDER_API_KEY,
    create_oci_httpx_auth,
    create_oci_openai_client,
    ensure_oci_openai_installed,
    get_oci_openai_base_url,
    get_oci_openai_headers,
    get_oci_openai_retry_kwargs,
    is_oci_openai_base_url,
    oci_genai_config_from_openai_client,
    validate_oci_openai_compatible_config,
)

SERVICE_ENDPOINT = "https://inference.generativeai.us-chicago-1.oci.oraclecloud.com"
COMPARTMENT_ID = "ocid1.compartment.oc1..aaaaaaaafakecompartment"
API_KEY_PROFILE = "APIKEY"
SESSION_PROFILE = "SESSION"

CHAT_COMPLETION_RESPONSE: Dict[str, Any] = {
    "id": "chatcmpl-test",
    "object": "chat.completion",
    "created": 0,
    "model": "openai.gpt-4.1",
    "choices": [
        {"index": 0, "message": {"role": "assistant", "content": "pong"}, "finish_reason": "stop"}
    ],
}


@pytest.fixture
def oci_config_file(tmp_path: Path) -> Path:
    """An OCI configuration file with an API key profile and a session token profile."""
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric import rsa

    private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    key_file = tmp_path / "oci_api_key.pem"
    key_file.write_bytes(
        private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.TraditionalOpenSSL,
            encryption_algorithm=serialization.NoEncryption(),
        )
    )
    token_file = tmp_path / "token"
    token_file.write_text("fake-session-token")
    config_file = tmp_path / "config"
    config_file.write_text(f"""[{API_KEY_PROFILE}]
user=ocid1.user.oc1..aaaaaaaafakeuser
fingerprint=aa:bb:cc:dd:ee:ff:00:11:22:33:44:55:66:77:88:99
tenancy=ocid1.tenancy.oc1..aaaaaaaafaketenancy
region=us-chicago-1
key_file={key_file}

[{SESSION_PROFILE}]
key_file={key_file}
security_token_file={token_file}
tenancy=ocid1.tenancy.oc1..aaaaaaaafaketenancy
region=us-chicago-1
""")
    return config_file


def _api_key_client_config(config_file: Path) -> OciClientConfigWithApiKey:
    return OciClientConfigWithApiKey(
        name="client_config",
        service_endpoint=SERVICE_ENDPOINT,
        auth_file_location=str(config_file),
        auth_profile=API_KEY_PROFILE,
    )


def _session_client_config(config_file: Path) -> OciClientConfigWithSecurityToken:
    return OciClientConfigWithSecurityToken(
        name="client_config",
        service_endpoint=SERVICE_ENDPOINT,
        auth_file_location=str(config_file),
        auth_profile=SESSION_PROFILE,
    )


def _llm_config(client_config: Any, **overrides: Any) -> OciGenAiConfig:
    # ``model_construct`` keeps these unit tests runnable when the test session patches the
    # constructors of LLM configurations to skip LLM-dependent tests (SKIP_LLM_TESTS=1): no
    # model is ever called here.
    fields: Dict[str, Any] = dict(
        name="oci_llm",
        model_id="openai.gpt-4.1",
        compartment_id=COMPARTMENT_ID,
        client_config=client_config,
    )
    fields.update(overrides)
    return OciGenAiConfig.model_construct(**fields)


def _mock_transport(recorded: list[httpx.Request]) -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        recorded.append(request)
        return httpx.Response(200, json=CHAT_COMPLETION_RESPONSE)

    return httpx.MockTransport(handler)


def test_base_url_is_the_openai_path_of_the_service_endpoint(oci_config_file: Path) -> None:
    llm_config = _llm_config(_api_key_client_config(oci_config_file))
    assert get_oci_openai_base_url(llm_config) == SERVICE_ENDPOINT + "/openai/v1"
    # A trailing slash in the endpoint does not double up
    llm_config.client_config.service_endpoint = SERVICE_ENDPOINT + "/"
    assert get_oci_openai_base_url(llm_config) == SERVICE_ENDPOINT + "/openai/v1"
    assert is_oci_openai_base_url(SERVICE_ENDPOINT + "/openai/v1/")
    assert not is_oci_openai_base_url("https://api.openai.com/v1")


def test_headers_carry_compartment_conversation_store_and_encoding(
    oci_config_file: Path,
) -> None:
    llm_config = _llm_config(_api_key_client_config(oci_config_file))
    assert get_oci_openai_headers(llm_config) == {
        COMPARTMENT_ID_HEADER: COMPARTMENT_ID,
        "Accept-Encoding": OCI_OPENAI_ACCEPT_ENCODING,
    }
    llm_config = _llm_config(
        _api_key_client_config(oci_config_file), conversation_store_id="ocid1.store.oc1..x"
    )
    assert get_oci_openai_headers(llm_config)[CONVERSATION_STORE_ID_HEADER] == "ocid1.store.oc1..x"


def test_auth_is_created_from_each_client_config_type(
    oci_config_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    oci_openai = ensure_oci_openai_installed()

    session_auth = create_oci_httpx_auth(_session_client_config(oci_config_file))
    assert isinstance(session_auth, oci_openai.OciSessionAuth)
    assert session_auth.profile_name == SESSION_PROFILE

    api_key_auth = create_oci_httpx_auth(_api_key_client_config(oci_config_file))
    assert isinstance(api_key_auth, oci_openai.OciUserPrincipalAuth)
    assert api_key_auth.profile_name == API_KEY_PROFILE

    # Principal-based authentications contact the OCI metadata service, so they are replaced
    class _FakeAuth(httpx.Auth):
        def __init__(self) -> None:
            self.created = True

    monkeypatch.setattr(oci_openai, "OciInstancePrincipalAuth", _FakeAuth)
    monkeypatch.setattr(oci_openai, "OciResourcePrincipalAuth", _FakeAuth)
    instance_principal = OciClientConfigWithInstancePrincipal(
        name="client_config", service_endpoint=SERVICE_ENDPOINT
    )
    resource_principal = OciClientConfigWithResourcePrincipal(
        name="client_config", service_endpoint=SERVICE_ENDPOINT
    )
    assert isinstance(create_oci_httpx_auth(instance_principal), _FakeAuth)
    assert isinstance(create_oci_httpx_auth(resource_principal), _FakeAuth)


def test_missing_oci_openai_package_raises_actionable_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "oci_openai", None)
    with pytest.raises(ImportError, match="pip install oci-openai"):
        ensure_oci_openai_installed()


def test_retry_policy_maps_to_client_retries_and_timeout() -> None:
    assert get_oci_openai_retry_kwargs(None) == {}
    assert get_oci_openai_retry_kwargs(RetryPolicy(max_attempts=5)) == {"max_retries": 5}
    assert get_oci_openai_retry_kwargs(RetryPolicy(max_attempts=1, request_timeout=30.0)) == {
        "max_retries": 1,
        "timeout": 30.0,
    }


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"serving_mode": ServingMode.DEDICATED}, "DEDICATED serving mode"),
        ({"provider": ModelProvider.COHERE}, "does not serve Cohere models"),
        ({"model_id": "cohere.command-a-03-2025"}, "does not serve Cohere models"),
        ({"api_type": OciAPIType.OPENAI_RESPONSES}, "does not support the OpenAI Responses API"),
    ],
)
def test_unsupported_configurations_are_rejected_early(
    oci_config_file: Path, overrides: Dict[str, Any], message: str
) -> None:
    llm_config = _llm_config(_api_key_client_config(oci_config_file), **overrides)
    with pytest.raises(NotImplementedError, match=message):
        validate_oci_openai_compatible_config(
            llm_config, runtime_name="TestRuntime", responses_api_supported=False
        )


def test_supported_configurations_pass_validation(oci_config_file: Path) -> None:
    client_config = _api_key_client_config(oci_config_file)
    for api_type in (OciAPIType.OCI, OciAPIType.OPENAI_CHAT_COMPLETIONS):
        validate_oci_openai_compatible_config(
            _llm_config(client_config, api_type=api_type),
            runtime_name="TestRuntime",
            responses_api_supported=False,
        )
    validate_oci_openai_compatible_config(
        _llm_config(client_config, api_type=OciAPIType.OPENAI_RESPONSES),
        runtime_name="TestRuntime",
        responses_api_supported=True,
    )


def _assert_signed_oci_request(request: httpx.Request, key_id_prefix: str) -> None:
    assert request.url.host == "inference.generativeai.us-chicago-1.oci.oraclecloud.com"
    assert request.url.path == "/openai/v1/chat/completions"
    authorization = request.headers["authorization"]
    assert authorization.startswith("Signature ")
    assert f'keyId="{key_id_prefix}' in authorization
    assert "x-content-sha256" in request.headers
    assert "date" in request.headers
    assert request.headers[COMPARTMENT_ID_HEADER] == COMPARTMENT_ID
    assert request.headers["accept-encoding"] == OCI_OPENAI_ACCEPT_ENCODING
    assert json.loads(request.content)["model"] == "openai.gpt-4.1"


def test_sync_client_signs_requests_with_api_key(oci_config_file: Path) -> None:
    recorded: list[httpx.Request] = []
    llm_config = _llm_config(
        _api_key_client_config(oci_config_file),
        retry_policy=RetryPolicy(max_attempts=0, request_timeout=12.5),
    )
    client = create_oci_openai_client(
        llm_config, is_async=False, transport=_mock_transport(recorded)
    )
    assert client.api_key == OCI_OPENAI_PLACEHOLDER_API_KEY
    assert str(client.base_url) == SERVICE_ENDPOINT + "/openai/v1/"
    assert client.max_retries == 0
    assert client.timeout == 12.5

    completion = client.chat.completions.create(
        model="openai.gpt-4.1", messages=[{"role": "user", "content": "ping"}]
    )
    assert completion.choices[0].message.content == "pong"
    assert len(recorded) == 1
    _assert_signed_oci_request(
        recorded[0], key_id_prefix="ocid1.tenancy.oc1..aaaaaaaafaketenancy/ocid1.user"
    )


@pytest.mark.anyio
async def test_async_client_signs_requests_with_session_token(oci_config_file: Path) -> None:
    recorded: list[httpx.Request] = []
    llm_config = _llm_config(_session_client_config(oci_config_file))
    client = create_oci_openai_client(
        llm_config, is_async=True, transport=_mock_transport(recorded)
    )

    completion = await client.chat.completions.create(
        model="openai.gpt-4.1", messages=[{"role": "user", "content": "ping"}]
    )
    assert completion.choices[0].message.content == "pong"
    assert len(recorded) == 1
    # Session tokens are referenced with the ST$ prefix in the signature key id
    _assert_signed_oci_request(recorded[0], key_id_prefix="ST$fake-session-token")


def test_config_is_rebuilt_from_an_oci_openai_client(oci_config_file: Path) -> None:
    llm_config = _llm_config(
        _session_client_config(oci_config_file), conversation_store_id="ocid1.store.oc1..x"
    )
    client = create_oci_openai_client(llm_config, is_async=True)

    rebuilt = oci_genai_config_from_openai_client(
        client, name="rebuilt", model_id="openai.gpt-4.1", api_type=OciAPIType.OPENAI_RESPONSES
    )
    assert rebuilt is not None
    assert rebuilt.model_id == "openai.gpt-4.1"
    assert rebuilt.compartment_id == COMPARTMENT_ID
    assert rebuilt.api_type == OciAPIType.OPENAI_RESPONSES
    assert rebuilt.conversation_store_id == "ocid1.store.oc1..x"
    assert isinstance(rebuilt.client_config, OciClientConfigWithSecurityToken)
    assert rebuilt.client_config.service_endpoint == SERVICE_ENDPOINT
    assert rebuilt.client_config.auth_profile == SESSION_PROFILE
    assert rebuilt.client_config.auth_file_location == str(oci_config_file)


def test_non_oci_client_is_not_converted() -> None:
    from openai import OpenAI

    client = OpenAI(api_key="sk-test", base_url="https://api.openai.com/v1")
    assert (
        oci_genai_config_from_openai_client(
            client, name="x", model_id="gpt-4.1", api_type=OciAPIType.OPENAI_CHAT_COMPLETIONS
        )
        is None
    )
