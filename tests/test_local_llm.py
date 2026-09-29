import json
from unittest.mock import MagicMock, patch

import pytest

from app.llm import LocalLLM


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


def test_chat_rejects_empty_messages():
    llm = LocalLLM()
    with pytest.raises(ValueError, match="Messages cannot be empty"):
        llm.chat([])


@patch("app.llm.local_llm.request.urlopen")
def test_chat_returns_text_response(mock_urlopen):
    mock_response = MagicMock()
    mock_response.read.return_value = json.dumps(
        {
            "message": {"role": "assistant", "content": "Bonjour."}
        }
    ).encode("utf-8")
    mock_response.__enter__.return_value = mock_response
    mock_urlopen.return_value = mock_response
    llm = LocalLLM()
    result = llm.chat([{"role": "user", "content": "Bonjour"}])
    assert result == {"role": "assistant", "content": "Bonjour."}


@patch("app.llm.local_llm.request.urlopen")
def test_chat_returns_tool_call(mock_urlopen):
    mock_response = MagicMock()
    mock_response.read.return_value = json.dumps(
        {
            "message": {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "function": {
                            "name": "list_files",
                            "arguments": {"path": ".", "max_results": 20},
                        }
                    }
                ],
            }
        }
    ).encode("utf-8")
    mock_response.__enter__.return_value = mock_response
    mock_urlopen.return_value = mock_response
    llm = LocalLLM()
    result = llm.chat(
        [{"role": "user", "content": "Liste les fichiers du projet."}],
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "list_files",
                    "description": "List project files.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {"type": "string"}
                        },
                    },
                },
            }
        ],
    )
    tool_call = result["tool_calls"][0]

    assert tool_call["function"]["name"] == "list_files"
    assert tool_call["function"]["arguments"]["path"] == "."


@patch("app.llm.local_llm.request.urlopen")
def test_chat_sends_tools_to_ollama(mock_urlopen):
    mock_response = MagicMock()
    mock_response.read.return_value = json.dumps(
        {
            "message": {"role": "assistant", "content": "OK"}
        }
    ).encode("utf-8")
    mock_response.__enter__.return_value = mock_response
    mock_urlopen.return_value = mock_response
    tools = [
        {
            "type": "function",
            "function": {
                "name": "list_files",
                "description": "List project files.",
                "parameters": {"type": "object", "properties": {}},
            },
        }
    ]
    llm = LocalLLM()
    llm.chat([{"role": "user", "content": "Test"}], tools=tools)
    req = mock_urlopen.call_args.args[0]
    payload = json.loads(req.data.decode("utf-8"))

    assert req.full_url == "http://localhost:11434/api/chat"
    assert payload["model"] == "qwen2.5-coder:14b"
    assert payload["messages"] == [{"role": "user", "content": "Test"}]
    assert payload["tools"] == tools
    assert payload["stream"] is False


@patch("app.llm.local_llm.request.urlopen")
def test_chat_updates_stats(mock_urlopen):
    mock_response = MagicMock()
    mock_response.read.return_value = json.dumps(
        {
            "message": {"role": "assistant", "content": "OK"},
            "prompt_eval_count": 10,
            "eval_count": 4,
        }
    ).encode("utf-8")
    mock_response.__enter__.return_value = mock_response
    mock_urlopen.return_value = mock_response
    llm = LocalLLM()
    llm.chat([{"role": "user", "content": "Test"}])
    assert llm.last_stats == {"prompt_tokens": 10, "completion_tokens": 4, "total_tokens": 14}
