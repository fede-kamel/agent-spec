# Copyright © 2026 Oracle and/or its affiliates.
#
# This software is under the Apache License 2.0
# (LICENSE-APACHE or http://www.apache.org/licenses/LICENSE-2.0) or Universal Permissive License
# (UPL) 1.0 (LICENSE-UPL or https://oss.oracle.com/licenses/upl), at your option.

"""Helpers to reach OCI Generative AI through its OpenAI-compatible API with the ``openai`` SDK.

The OCI Generative AI service exposes OpenAI-compatible ``chat/completions`` and ``responses``
endpoints under ``<service_endpoint>/openai/v1``. Requests must carry an OCI request signature
and the target compartment in the ``opc-compartment-id`` header. The request signing is provided
by the ``oci-openai`` package, which offers ``httpx.Auth`` implementations for the four OCI
authentication types modelled by the Agent Spec ``OciClientConfig`` components.

These helpers are shared by the runtime adapters built on the ``openai`` SDK (AutoGen, Microsoft
Agent Framework and OpenAI Agents), so that an ``OciGenAiConfig`` can be executed by those
runtimes without provider-specific code in each adapter.
"""

import os
from types import ModuleType
from typing import TYPE_CHECKING, Any, Dict, Optional, Union, cast

from pyagentspec.llms.ociclientconfig import (
    OciClientConfig,
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

if TYPE_CHECKING:
    import httpx
    from openai import AsyncOpenAI, OpenAI

OCI_OPENAI_PLACEHOLDER_API_KEY = "<NOTUSED>"
"""API key placeholder: OCI requests are authenticated with a request signature, not a key."""

OCI_OPENAI_BASE_PATH = "/openai/v1"
"""Path of the OpenAI-compatible API under the OCI Generative AI service endpoint."""

COMPARTMENT_ID_HEADER = "opc-compartment-id"
CONVERSATION_STORE_ID_HEADER = "opc-conversation-store-id"

OCI_OPENAI_ACCEPT_ENCODING = "gzip, deflate"
"""Accepted response encodings.

The OpenAI-compatible endpoint negotiates ``zstd`` when the client advertises it and compresses
every streamed chunk as an independent zstd frame, which the ``httpx`` decoder rejects
("cannot use a decompressobj multiple times"). gzip streams are decoded correctly.
"""

OCI_OPENAI_INSTALL_HINT = (
    "The `oci-openai` package is required to run OCI Generative AI models through the "
    "OpenAI-compatible API. Install it with `pip install oci-openai` "
    "(or `pip install pyagentspec[oci]`)."
)


def ensure_oci_openai_installed() -> ModuleType:
    """Return the ``oci_openai`` module, raising an actionable error when it is missing."""
    try:
        import oci_openai  # type: ignore
    except ImportError as exc:
        raise ImportError(OCI_OPENAI_INSTALL_HINT) from exc
    return oci_openai


def get_oci_service_endpoint(llm_config: OciGenAiConfig) -> str:
    """Return the OCI Generative AI service endpoint of the configuration, without trailing slash."""
    return llm_config.client_config.service_endpoint.rstrip("/")


def get_oci_openai_base_url(llm_config: OciGenAiConfig) -> str:
    """Return the base URL of the OpenAI-compatible API for the configuration."""
    return get_oci_service_endpoint(llm_config) + OCI_OPENAI_BASE_PATH


def uses_responses_api(llm_config: OciGenAiConfig) -> bool:
    """Whether the configuration asks for the OpenAI Responses API rather than chat completions."""
    return llm_config.api_type == OciAPIType.OPENAI_RESPONSES


def validate_oci_openai_compatible_config(
    llm_config: OciGenAiConfig,
    *,
    runtime_name: str,
    responses_api_supported: bool,
) -> None:
    """Fail early for configurations the OpenAI-compatible API of OCI cannot serve.

    Parameters
    ----------
    llm_config:
        The OCI Generative AI configuration to execute.
    runtime_name:
        Name of the runtime, used in error messages.
    responses_api_supported:
        Whether the runtime can execute the OpenAI Responses API.

    Raises
    ------
    NotImplementedError
        For dedicated serving mode, Cohere models (served by the native OCI API only) and the
        Responses API when the runtime does not support it.
    """
    if llm_config.serving_mode == ServingMode.DEDICATED:
        raise NotImplementedError(
            f"{runtime_name} runs OCI Generative AI models through the OpenAI-compatible API, "
            "which does not support the DEDICATED serving mode. Use the ON_DEMAND serving mode, "
            "or a runtime using the native OCI API (WayFlow, LangGraph)."
        )
    is_cohere_model = (
        llm_config.provider == ModelProvider.COHERE
        or llm_config.model_id.lower().startswith("cohere.")
    )
    if is_cohere_model:
        raise NotImplementedError(
            f"{runtime_name} runs OCI Generative AI models through the OpenAI-compatible API, "
            f"which does not serve Cohere models such as '{llm_config.model_id}'. Use a runtime "
            "using the native OCI API (WayFlow, LangGraph) for Cohere models."
        )
    if uses_responses_api(llm_config) and not responses_api_supported:
        raise NotImplementedError(
            f"{runtime_name} does not support the OpenAI Responses API; set the `api_type` of "
            f"the OCI Generative AI configuration '{llm_config.name}' to "
            f"'{OciAPIType.OPENAI_CHAT_COMPLETIONS.value}' or '{OciAPIType.OCI.value}'."
        )


def create_oci_httpx_auth(client_config: OciClientConfig) -> "httpx.Auth":
    """Create the ``httpx`` authentication signing requests for an OCI client configuration."""
    oci_openai = ensure_oci_openai_installed()
    if isinstance(client_config, OciClientConfigWithSecurityToken):
        return oci_openai.OciSessionAuth(  # type: ignore[no-any-return]
            config_file=os.path.expanduser(client_config.auth_file_location),
            profile_name=client_config.auth_profile,
        )
    if isinstance(client_config, OciClientConfigWithApiKey):
        return oci_openai.OciUserPrincipalAuth(  # type: ignore[no-any-return]
            config_file=os.path.expanduser(client_config.auth_file_location),
            profile_name=client_config.auth_profile,
        )
    if isinstance(client_config, OciClientConfigWithInstancePrincipal):
        return oci_openai.OciInstancePrincipalAuth()  # type: ignore[no-any-return]
    if isinstance(client_config, OciClientConfigWithResourcePrincipal):
        return oci_openai.OciResourcePrincipalAuth()  # type: ignore[no-any-return]
    raise NotImplementedError(
        f"Unsupported OCI client configuration of type {type(client_config).__name__}"
    )


def get_oci_openai_headers(llm_config: OciGenAiConfig) -> Dict[str, str]:
    """Return the headers every request to the OpenAI-compatible API of OCI must carry."""
    headers = {
        COMPARTMENT_ID_HEADER: llm_config.compartment_id,
        "Accept-Encoding": OCI_OPENAI_ACCEPT_ENCODING,
    }
    if llm_config.conversation_store_id:
        headers[CONVERSATION_STORE_ID_HEADER] = llm_config.conversation_store_id
    return headers


def get_oci_openai_retry_kwargs(retry_policy: Optional[RetryPolicy]) -> Dict[str, Any]:
    """Map an Agent Spec retry policy onto the ``openai`` client retry arguments."""
    if retry_policy is None:
        return {}
    retry_kwargs: Dict[str, Any] = {"max_retries": retry_policy.max_attempts}
    if retry_policy.request_timeout is not None:
        retry_kwargs["timeout"] = retry_policy.request_timeout
    return retry_kwargs


def create_oci_openai_httpx_client(
    llm_config: OciGenAiConfig,
    *,
    is_async: bool,
    **httpx_client_kwargs: Any,
) -> Union["httpx.Client", "httpx.AsyncClient"]:
    """Create the signing ``httpx`` client used by the ``openai`` SDK for an OCI configuration.

    The client signs every request with the OCI credentials of the configuration and carries the
    compartment (and conversation store) headers. Extra keyword arguments are passed to the
    ``httpx`` client constructor.
    """
    from openai import DefaultAsyncHttpxClient, DefaultHttpxClient

    client_class = DefaultAsyncHttpxClient if is_async else DefaultHttpxClient
    return client_class(  # type: ignore[no-any-return]
        auth=create_oci_httpx_auth(llm_config.client_config),
        headers=get_oci_openai_headers(llm_config),
        **httpx_client_kwargs,
    )


def create_oci_openai_client(
    llm_config: OciGenAiConfig,
    *,
    is_async: bool,
    **httpx_client_kwargs: Any,
) -> Union["OpenAI", "AsyncOpenAI"]:
    """Create an ``openai`` client bound to the OpenAI-compatible API of OCI Generative AI.

    Parameters
    ----------
    llm_config:
        The OCI Generative AI configuration to execute.
    is_async:
        Whether to create an ``AsyncOpenAI`` (``True``) or an ``OpenAI`` (``False``) client.
    httpx_client_kwargs:
        Extra keyword arguments for the underlying ``httpx`` client (for instance a transport).
    """
    from openai import AsyncOpenAI, OpenAI

    client_kwargs: Dict[str, Any] = {
        "api_key": OCI_OPENAI_PLACEHOLDER_API_KEY,
        "base_url": get_oci_openai_base_url(llm_config),
        **get_oci_openai_retry_kwargs(llm_config.retry_policy),
    }
    http_client = create_oci_openai_httpx_client(
        llm_config, is_async=is_async, **httpx_client_kwargs
    )
    if is_async:
        return AsyncOpenAI(http_client=cast("httpx.AsyncClient", http_client), **client_kwargs)
    return OpenAI(http_client=cast("httpx.Client", http_client), **client_kwargs)


def is_oci_openai_base_url(base_url: str) -> bool:
    """Whether a base URL points to the OpenAI-compatible API of OCI Generative AI."""
    return base_url.rstrip("/").endswith(OCI_OPENAI_BASE_PATH)


def oci_client_config_from_httpx_auth(
    auth: Any, *, service_endpoint: str, name: str
) -> Optional[OciClientConfig]:
    """Rebuild the Agent Spec client configuration from an ``oci-openai`` authentication.

    Returns ``None`` when the authentication is not one created by ``oci-openai``.
    """
    try:
        import oci_openai  # type: ignore
    except ImportError:
        return None
    if isinstance(auth, oci_openai.OciSessionAuth):
        return OciClientConfigWithSecurityToken(
            name=name,
            service_endpoint=service_endpoint,
            auth_file_location=auth.config_file,
            auth_profile=auth.profile_name,
        )
    if isinstance(auth, oci_openai.OciUserPrincipalAuth):
        return OciClientConfigWithApiKey(
            name=name,
            service_endpoint=service_endpoint,
            auth_file_location=auth.config_file,
            auth_profile=auth.profile_name,
        )
    if isinstance(auth, oci_openai.OciInstancePrincipalAuth):
        return OciClientConfigWithInstancePrincipal(name=name, service_endpoint=service_endpoint)
    if isinstance(auth, oci_openai.OciResourcePrincipalAuth):
        return OciClientConfigWithResourcePrincipal(name=name, service_endpoint=service_endpoint)
    return None


def oci_genai_config_from_openai_client(
    client: Union["OpenAI", "AsyncOpenAI"],
    *,
    name: str,
    model_id: str,
    api_type: OciAPIType,
) -> Optional[OciGenAiConfig]:
    """Rebuild an ``OciGenAiConfig`` from an ``openai`` client created for OCI Generative AI.

    Returns ``None`` when the client does not target the OpenAI-compatible API of OCI or when
    its authentication cannot be mapped back to an Agent Spec client configuration.
    """
    base_url = str(client.base_url).rstrip("/")
    if not is_oci_openai_base_url(base_url):
        return None
    http_client = getattr(client, "_client", None)
    headers = getattr(http_client, "headers", None) or {}
    compartment_id = headers.get(COMPARTMENT_ID_HEADER)
    if not compartment_id:
        return None
    service_endpoint = base_url[: -len(OCI_OPENAI_BASE_PATH)]
    client_config = oci_client_config_from_httpx_auth(
        getattr(http_client, "auth", None),
        service_endpoint=service_endpoint,
        name=f"{name}_client_config",
    )
    if client_config is None:
        return None
    return OciGenAiConfig(
        name=name,
        model_id=model_id,
        compartment_id=compartment_id,
        client_config=client_config,
        api_type=api_type,
        conversation_store_id=headers.get(CONVERSATION_STORE_ID_HEADER),
    )
