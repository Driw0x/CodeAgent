import json
from unittest.mock import MagicMock, patch

import pytest

from app.llm.local_llm import LocalLLM


def test_default_model():
    llm = LocalLLM()

    assert llm.model == "qwen2.5-coder:14b"


def test_generate_rejects_empty_prompt():
    llm = LocalLLM()

    with pytest.raises(ValueError, match="Prompt cannot be empty"):
        llm.generate("")


@patch("app.llm.local_llm.request.urlopen")
def test_generate_returns_response(mock_urlopen):
    mock_response = MagicMock()
    mock_response.read.return_value = json.dumps(
        {"response": "La fonction retourne la somme de a et b."}
    ).encode("utf-8")
    mock_response.__enter__.return_value = mock_response
    mock_urlopen.return_value = mock_response

    llm = LocalLLM()
    result = llm.generate("Que fait cette fonction ?")

    assert result == "La fonction retourne la somme de a et b."


@patch("app.llm.local_llm.request.urlopen")
def test_generate_sends_selected_model(mock_urlopen):
    mock_response = MagicMock()
    mock_response.read.return_value = b'{"response": "OK"}'
    mock_response.__enter__.return_value = mock_response
    mock_urlopen.return_value = mock_response

    llm = LocalLLM()
    llm.generate("Test")

    req = mock_urlopen.call_args.args[0]
    payload = json.loads(req.data.decode("utf-8"))

    assert payload["model"] == "qwen2.5-coder:14b"
    assert payload["stream"] is False
    assert payload["options"]["temperature"] == 0.0
    assert payload["options"]["seed"] == 42
    assert payload["options"]["num_predict"] == 512