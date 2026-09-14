# Copyright © 2026 Oracle and/or its affiliates.
#
# This software is under the Apache License 2.0
# (LICENSE-APACHE or http://www.apache.org/licenses/LICENSE-2.0) or Universal Permissive License
# (UPL) 1.0 (LICENSE-UPL or https://oss.oracle.com/licenses/upl), at your option.
"""OCI Generative AI configurations run on AutoGen through the OpenAI-compatible API (no network)."""

import json
from pathlib import Path
from typing import Any, Dict, List

import httpx
import pytest

from pyagentspec.llms.ociclientconfig import OciClientConfigWithApiKey
from pyagentspec.llms.ocigenaiconfig import (
    ModelProvider,
    OciAPIType,
    OciGenAiConfig,
    ServingMode,
)
from pyagentspec.retrypolicy import RetryPolicy

pytest.importorskip("oci_openai")

SERVICE_ENDPOINT = "https://inference.generativeai.us-chicago-1.oci.oraclecloud.com"
COMPARTMENT_ID = "ocid1.compartment.oc1..aaaaaaaafakecompartment"
PROFILE = "APIKEY"

CHAT_COMPLETION_RESPONSE: Dict[str, Any] = {
    "id": "chatcmpl-test",
    "object": "chat.completion",
    "created": 0,
    "model": "openai.gpt-4.1",
    "choices": [
        {"index": 0, "message": {"role": "assistant", "content": "pong"}, "finish_reason": "stop"}
    ],
    "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
}


@pytest.fixture
def oci_config_file(tmp_path: Path) -> Path:
    """An OCI configuration file with an API key profile backed by a generated key."""
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
    config_file = tmp_path / "config"
    config_file.write_text(f"""[{PROFILE}]
user=ocid1.user.oc1..aaaaaaaafakeuser
fingerprint=aa:bb:cc:dd:ee:ff:00:11:22:33:44:55:66:77:88:99
tenancy=ocid1.tenancy.oc1..aaaaaaaafaketenancy
region=us-chicago-1
key_file={key_file}
""")
    return config_file


def _client_config(config_file: Path) -> OciClientConfigWithApiKey:
    return OciClientConfigWithApiKey(
        name="client_config",
        service_endpoint=SERVICE_ENDPOINT,
        auth_file_location=str(config_file),
        auth_profile=PROFILE,
    )


def _llm_config(config_file: Path, **overrides: Any) -> OciGenAiConfig:
    # ``model_construct`` keeps these tests runnable when the test session patches the
    # constructors of LLM configurations to skip LLM-dependent tests (SKIP_LLM_TESTS=1): no
    # model is ever called here.
    fields: Dict[str, Any] = dict(
        name="oci_llm",
        model_id="openai.gpt-4.1",
        compartment_id=COMPARTMENT_ID,
        client_config=_client_config(config_file),
    )
    fields.update(overrides)
    return OciGenAiConfig.model_construct(**fields)


def _convert(llm_config: OciGenAiConfig) -> Any:
    from pyagentspec.adapters.autogen._autogenconverter import AgentSpecToAutogenConverter

    return AgentSpecToAutogenConverter()._llm_convert_to_autogen(
        llm_config, tool_registry={}, converted_components={}
    )


def test_oci_config_converts_to_a_signed_openai_client(oci_config_file: Path) -> None:
    import oci_openai
    from autogen_ext.models.openai import OpenAIChatCompletionClient

    from pyagentspec.adapters._oci_openai_common import OCI_OPENAI_PLACEHOLDER_API_KEY
    from pyagentspec.adapters.autogen._types import AutogenModelFamily

    client = _convert(_llm_config(oci_config_file))

    assert isinstance(client, OpenAIChatCompletionClient)
    assert client.model_info["family"] == AutogenModelFamily.UNKNOWN
    assert client.model_info["function_calling"] is True
    openai_client = client._client
    assert str(openai_client.base_url) == SERVICE_ENDPOINT + "/openai/v1/"
    assert openai_client.api_key == OCI_OPENAI_PLACEHOLDER_API_KEY
    http_client = openai_client._client
    assert http_client.headers["opc-compartment-id"] == COMPARTMENT_ID
    assert http_client.headers["accept-encoding"] == "gzip, deflate"
    assert isinstance(http_client.auth, oci_openai.OciUserPrincipalAuth)
    assert http_client.auth.profile_name == PROFILE
    # The signing client is not part of the serializable configuration
    assert "http_client" not in client.dump_component().config


def test_retry_policy_maps_to_client_retries_and_timeout(oci_config_file: Path) -> None:
    client = _convert(
        _llm_config(oci_config_file, retry_policy=RetryPolicy(max_attempts=0, request_timeout=12.5))
    )
    assert client._client.max_retries == 0
    assert client._client.timeout == 12.5


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"serving_mode": ServingMode.DEDICATED}, "DEDICATED serving mode"),
        ({"provider": ModelProvider.COHERE}, "does not serve Cohere models"),
        ({"model_id": "cohere.command-a-03-2025"}, "does not serve Cohere models"),
        ({"api_type": OciAPIType.OPENAI_RESPONSES}, "does not support the OpenAI Responses API"),
    ],
)
def test_unsupported_oci_configs_are_rejected_before_building_a_client(
    oci_config_file: Path,
    monkeypatch: pytest.MonkeyPatch,
    overrides: Dict[str, Any],
    message: str,
) -> None:
    from pyagentspec.adapters.autogen import _autogenconverter

    def _must_not_be_called(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("no client should be built for an unsupported configuration")

    monkeypatch.setattr(_autogenconverter, "create_oci_openai_httpx_client", _must_not_be_called)
    with pytest.raises(NotImplementedError, match=message):
        _convert(_llm_config(oci_config_file, **overrides))


@pytest.mark.anyio
async def test_autogen_requests_are_signed_for_oci(
    oci_config_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from autogen_core.models import UserMessage

    from pyagentspec.adapters import _oci_openai_common
    from pyagentspec.adapters.autogen import _autogenconverter

    recorded: List[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        recorded.append(request)
        return httpx.Response(200, json=CHAT_COMPLETION_RESPONSE)

    def create_httpx_client_with_mock_transport(
        llm_config: OciGenAiConfig, *, is_async: bool, **kwargs: Any
    ) -> Any:
        return _oci_openai_common.create_oci_openai_httpx_client(
            llm_config, is_async=is_async, transport=httpx.MockTransport(handler), **kwargs
        )

    monkeypatch.setattr(
        _autogenconverter,
        "create_oci_openai_httpx_client",
        create_httpx_client_with_mock_transport,
    )
    client = _convert(_llm_config(oci_config_file))

    result = await client.create([UserMessage(content="ping", source="user")])

    assert result.content == "pong"
    assert len(recorded) == 1
    request = recorded[0]
    assert request.url.host == "inference.generativeai.us-chicago-1.oci.oraclecloud.com"
    assert request.url.path == "/openai/v1/chat/completions"
    assert request.headers["authorization"].startswith("Signature ")
    assert (
        'keyId="ocid1.tenancy.oc1..aaaaaaaafaketenancy/ocid1.user'
        in request.headers["authorization"]
    )
    assert request.headers["opc-compartment-id"] == COMPARTMENT_ID
    assert request.headers["accept-encoding"] == "gzip, deflate"
    assert "x-content-sha256" in request.headers
    assert json.loads(request.content)["model"] == "openai.gpt-4.1"


def test_oci_client_converts_back_to_oci_config(oci_config_file: Path) -> None:
    """Round trip Agent Spec -> AutoGen -> Agent Spec.

    Builds the configuration with its constructor, so the test is skipped by the LLM guard of
    the test session (SKIP_LLM_TESTS=1) and runs in a local environment.
    """
    from pyagentspec.adapters.autogen._agentspecconverter import AutogenToAgentSpecConverter

    llm_config = OciGenAiConfig(
        name="oci_llm",
        model_id="openai.gpt-4.1",
        compartment_id=COMPARTMENT_ID,
        client_config=_client_config(oci_config_file),
        api_type=OciAPIType.OPENAI_CHAT_COMPLETIONS,
        conversation_store_id="ocid1.store.oc1..x",
    )
    client = _convert(llm_config)

    rebuilt = AutogenToAgentSpecConverter()._llm_convert_to_agentspec(client)

    assert isinstance(rebuilt, OciGenAiConfig)
    assert rebuilt.model_id == llm_config.model_id
    assert rebuilt.compartment_id == COMPARTMENT_ID
    assert rebuilt.api_type == OciAPIType.OPENAI_CHAT_COMPLETIONS
    assert rebuilt.conversation_store_id == "ocid1.store.oc1..x"
    assert isinstance(rebuilt.client_config, OciClientConfigWithApiKey)
    assert rebuilt.client_config.service_endpoint == SERVICE_ENDPOINT
    assert rebuilt.client_config.auth_profile == PROFILE
    assert rebuilt.client_config.auth_file_location == str(oci_config_file)
