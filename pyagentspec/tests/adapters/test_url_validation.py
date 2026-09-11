# Copyright © 2026 Oracle and/or its affiliates.
#
# This software is under the Apache License 2.0
# (LICENSE-APACHE or http://www.apache.org/licenses/LICENSE-2.0) or Universal Permissive License
# (UPL) 1.0 (LICENSE-UPL or https://oss.oracle.com/licenses/upl), at your option.
"""URL allow-list matching used by RemoteTool and ApiNode adapters."""

from typing import List

import pytest

from pyagentspec.adapters._url_validation import (
    _matches_allow_list_entry,
    validate_url_against_allow_list,
)


@pytest.mark.parametrize(
    "url, pattern",
    [
        # Same path, with or without a trailing slash
        ("https://h.example/api", "https://h.example/api"),
        ("https://h.example/api/", "https://h.example/api"),
        ("https://h.example/api/", "https://h.example/api/"),
        # Sub-paths of the allowed path
        ("https://h.example/api/x", "https://h.example/api"),
        ("https://h.example/api/x/y", "https://h.example/api/"),
        ("https://h.example/orders/123/items", "https://h.example/orders/"),
        # A pattern without path allows every path on the origin
        ("https://h.example/anything/at/all", "https://h.example"),
        ("https://h.example/anything/at/all", "https://h.example/"),
        ("https://h.example", "https://h.example"),
        # Query parameters and fragments are not part of the match
        ("https://h.example/api/x?token=1#frag", "https://h.example/api"),
        # Default ports and host case are normalized
        ("https://h.example:443/api/x", "https://h.example/api"),
        ("https://H.EXAMPLE/api/x", "https://h.example/api"),
    ],
)
def test_urls_inside_the_allowed_path_match(url: str, pattern: str) -> None:
    assert _matches_allow_list_entry(url, pattern)
    validate_url_against_allow_list(url, [pattern])


@pytest.mark.parametrize(
    "url, pattern",
    [
        # The allowed path is only a string prefix of the requested path
        ("https://h.example/apiX/evil", "https://h.example/api"),
        ("https://h.example/api-evil", "https://h.example/api"),
        ("https://h.example/api.evil/x", "https://h.example/api"),
        ("https://h.example/orders123", "https://h.example/orders"),
        # A different path
        ("https://h.example/customers/123", "https://h.example/orders/"),
        ("https://h.example/", "https://h.example/api"),
        # A pattern ending with "/" denotes the subtree under that directory only
        ("https://h.example/api", "https://h.example/api/"),
        # Path traversal, plain and percent-encoded, does not escape the allowed path
        ("https://h.example/api/../admin", "https://h.example/api"),
        ("https://h.example/api/%2e%2e/admin", "https://h.example/api"),
        # Scheme, host, port and userinfo must match exactly
        ("http://h.example/api/x", "https://h.example/api"),
        ("https://h.example:8443/api/x", "https://h.example/api"),
        ("https://h.example.evil/api/x", "https://h.example/api"),
        ("https://evil.example/h.example/api/x", "https://h.example/api"),
        ("https://user@h.example/api/x", "https://h.example/api"),
    ],
)
def test_urls_outside_the_allowed_path_do_not_match(url: str, pattern: str) -> None:
    assert not _matches_allow_list_entry(url, pattern)
    with pytest.raises(ValueError, match="not in allowed list"):
        validate_url_against_allow_list(url, [pattern])


def test_any_matching_entry_allows_the_url() -> None:
    allow_list: List[str] = ["https://other.example/", "https://h.example/api"]
    validate_url_against_allow_list("https://h.example/api/x", allow_list)
    with pytest.raises(ValueError, match="not in allowed list"):
        validate_url_against_allow_list("https://h.example/apix", allow_list)


def test_no_allow_list_accepts_any_url() -> None:
    validate_url_against_allow_list("https://anywhere.example/path", None)
