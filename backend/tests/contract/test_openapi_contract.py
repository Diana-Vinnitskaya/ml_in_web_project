from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


CONTRACT_PATH = (
    Path(__file__).resolve().parents[3]
    / "contracts"
    / "openapi.yaml"
)


def load_contract() -> dict[str, Any]:
    return yaml.safe_load(CONTRACT_PATH.read_text(encoding="utf-8"))


def resolve_contract_response(
    contract: dict[str, Any],
    response_spec: dict[str, Any],
) -> dict[str, Any]:
    ref = response_spec.get("$ref")
    if not ref:
        return response_spec

    resolved: Any = contract
    for segment in ref.removeprefix("#/").split("/"):
        resolved = resolved[segment]
    return resolved


def test_openapi_contract_metadata_matches_generated_schema(
    client_with_fake_classifier,
) -> None:
    contract = load_contract()
    schema = client_with_fake_classifier.app.openapi()

    assert schema["info"] == contract["info"]
    assert schema.get("servers") == contract["servers"]


def test_openapi_contract_public_paths_and_operations_match(
    client_with_fake_classifier,
) -> None:
    contract = load_contract()
    schema = client_with_fake_classifier.app.openapi()

    assert set(schema["paths"]) == set(contract["paths"])

    for path, contract_operations in contract["paths"].items():
        generated_operations = schema["paths"][path]

        assert set(generated_operations) == set(contract_operations)

        for method, contract_operation in contract_operations.items():
            generated_operation = generated_operations[method]

            assert generated_operation["summary"] == contract_operation["summary"]
            assert generated_operation["operationId"] == contract_operation["operationId"]
            assert generated_operation["tags"] == contract_operation["tags"]

            if "servers" in contract_operation:
                assert generated_operation.get("servers") == contract_operation["servers"]

            if "requestBody" in contract_operation:
                contract_request = contract_operation["requestBody"]["content"][
                    "application/json"
                ]["schema"]
                generated_request = generated_operation["requestBody"]["content"][
                    "application/json"
                ]["schema"]
                assert generated_request == contract_request
            else:
                assert "requestBody" not in generated_operation

            assert set(generated_operation["responses"]) == set(
                contract_operation["responses"],
            )

            for status_code, contract_response in contract_operation["responses"].items():
                resolved_contract_response = resolve_contract_response(
                    contract,
                    contract_response,
                )
                generated_response = generated_operation["responses"][status_code]

                if status_code == "422":
                    assert generated_response["description"] in {
                        resolved_contract_response["description"],
                        "Validation Error",
                    }
                    assert "application/json" in generated_response.get("content", {})
                    continue

                assert (
                    generated_response["description"]
                    == resolved_contract_response["description"]
                )

                contract_content = resolved_contract_response.get("content", {})
                generated_content = generated_response.get("content", {})

                assert set(generated_content) == set(contract_content)

                for media_type, media_spec in contract_content.items():
                    assert (
                        generated_content[media_type]["schema"]
                        == media_spec["schema"]
                    )
