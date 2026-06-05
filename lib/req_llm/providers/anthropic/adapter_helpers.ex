defmodule ReqLLM.Providers.Anthropic.AdapterHelpers do
  @moduledoc """
  Shared helper functions for Anthropic model adapters (Bedrock, Vertex).

  These functions are NOT used by the native Anthropic provider - they are
  specific to adapters that wrap Anthropic's API in other platforms.
  """

  @doc """
  Conditionally add a parameter to a map if the value is not nil.
  """
  def maybe_add_param(body, _key, nil), do: body
  def maybe_add_param(body, key, value), do: Map.put(body, key, value)

  @doc """
  Prepare context and options for `:object` operations using structured output.

  Creates a synthetic `structured_output` tool and forces tool choice to use it,
  leveraging Claude's tool-calling for structured JSON output.

  ## Strict mode

  When the caller passes `anthropic_structured_output_mode: :tool_strict` (or
  `:json_schema`), the synthetic tool is marked `strict: true` so Claude's
  grammar-constrained decoding guarantees the response matches the schema. The
  default (`:auto`, or the option omitted) keeps best-effort tool calling.

  Strict mode exists because, without it, Claude intermittently double-encodes
  nested array and object fields as JSON strings (for example
  `"options": "[\\"a\\", \\"b\\"]"` instead of `"options": ["a", "b"]`), which
  downstream schema validation then rejects. See
  [anthropics/claude-agent-sdk-python#510](https://github.com/anthropics/claude-agent-sdk-python/issues/510).

  It is opt-in rather than the default because the underlying provider feature
  may be gated. On Google Vertex AI it requires the `structured_outputs` partner-model
  feature to be allow-listed by the GCP org policy
  (`constraints/vertexai.allowedPartnerModelFeatures`); otherwise the request is
  rejected with HTTP 400 `FAILED_PRECONDITION`.

  ## Schema reduction (strict mode only)

  Strict (grammar-constrained) decoding only supports a subset of JSON Schema, so
  the compiled schema is reduced before it is sent:

    * `strip_constraints_recursive/1` removes unsupported keywords
      (`minimum`, `maximum`, `minLength`, `maxLength`, `minItems`, `maxItems`).
      Array length is therefore no longer enforced at the provider; callers that
      need an exact count must re-validate it themselves.
    * `enforce_strict_schema_requirements/1` marks every property required and
      sets `additionalProperties: false`, both mandatory for strict mode.
  """
  def prepare_structured_output_context(context, opts) do
    compiled_schema = Keyword.fetch!(opts, :compiled_schema)
    structured_output_tool = build_structured_output_tool(compiled_schema, strict?(opts))

    existing_tools = Map.get(context, :tools, [])
    updated_context = Map.put(context, :tools, [structured_output_tool | existing_tools])

    updated_opts =
      opts
      |> Keyword.put(:tools, [structured_output_tool | Keyword.get(opts, :tools, [])])
      |> Keyword.put(:tool_choice, %{type: "tool", name: "structured_output"})

    {updated_context, updated_opts}
  end

  defp build_structured_output_tool(compiled_schema, true) do
    schema =
      compiled_schema.schema
      |> ReqLLM.Schema.to_json()
      |> strip_constraints_recursive()
      |> enforce_strict_schema_requirements()

    ReqLLM.Tool.new!(
      name: "structured_output",
      description: "Generate structured output matching the provided schema",
      parameter_schema: schema,
      strict: true,
      callback: fn _args -> {:ok, "structured output generated"} end
    )
  end

  defp build_structured_output_tool(compiled_schema, false) do
    ReqLLM.Tool.new!(
      name: "structured_output",
      description: "Generate structured output matching the provided schema",
      parameter_schema: compiled_schema.schema,
      callback: fn _args -> {:ok, "structured output generated"} end
    )
  end

  defp strict?(opts) do
    provider_opts = get_opt(opts, :provider_options) || []

    mode =
      get_opt(opts, :anthropic_structured_output_mode) ||
        get_opt(provider_opts, :anthropic_structured_output_mode) ||
        :auto

    mode in [:tool_strict, :json_schema]
  end

  defp get_opt(opts, key) when is_list(opts), do: Keyword.get(opts, key)
  defp get_opt(opts, key) when is_map(opts), do: Map.get(opts, key)
  defp get_opt(_opts, _key), do: nil

  @doc """
  Add extended thinking configuration to request body if enabled.

  Extended thinking doesn't work when tool_choice forces a specific tool.
  See: https://docs.claude.com/en/docs/build-with-claude/extended-thinking
  """
  def maybe_add_thinking(body, opts) do
    # Check if additional_model_request_fields has thinking config
    thinking_config =
      get_in(opts, [:provider_options, :additional_model_request_fields, :thinking])

    tool_choice = opts[:tool_choice]

    # Extended thinking doesn't work when tool_choice forces a specific tool
    forced_tool_choice? =
      case tool_choice do
        %{type: "tool", name: _} -> true
        %{"type" => "tool", "name" => _} -> true
        _ -> false
      end

    case thinking_config do
      %{type: "enabled", budget_tokens: budget} when not forced_tool_choice? ->
        Map.put(body, :thinking, %{type: "enabled", budget_tokens: budget})

      _ ->
        body
    end
  end

  @doc """
  Extract structured output from tool calls in response.

  Used for :object operations to get the final structured output.
  """
  def extract_and_set_object(response) do
    extracted_object =
      response
      |> ReqLLM.Response.tool_calls()
      |> ReqLLM.ToolCall.find_args("structured_output")

    %{response | object: extracted_object}
  end

  @doc """
  Extract stub tool definitions from messages when tools are needed but none provided.

  Bedrock and Azure are strict about tool validation in multi-turn conversations.
  If the conversation history contains tool_use or tool_result blocks, the API
  requires corresponding tool definitions. This function extracts tool names
  from the messages and creates minimal stub definitions.
  """
  @spec extract_stub_tools_from_messages(map()) :: [map()]
  def extract_stub_tools_from_messages(body) do
    messages = Map.get(body, :messages, [])

    tool_names =
      messages
      |> Enum.flat_map(fn msg ->
        case msg do
          %{content: content} when is_list(content) ->
            content
            |> Enum.filter(fn
              %{type: "tool_use", name: _} -> true
              %{"type" => "tool_use", "name" => _} -> true
              %{type: "tool_result", tool_use_id: _} -> true
              %{"type" => "tool_result", "toolUseId" => _} -> true
              _ -> false
            end)
            |> Enum.map(fn
              %{name: name} -> name
              %{"name" => name} -> name
              _ -> "__tool_result_placeholder__"
            end)

          _ ->
            []
        end
      end)
      |> Enum.uniq()

    Enum.map(tool_names, fn name ->
      %{
        name: name,
        description: "Tool stub for multi-turn conversation",
        input_schema: %{type: "object", properties: %{}}
      }
    end)
  end

  @doc """
  Recursively remove JSON Schema keywords that strict (grammar-constrained)
  decoding does not support.

  Strips `minimum`, `maximum`, `minLength`, `maxLength`, `minItems`, and
  `maxItems` from the schema and every nested `properties`/`items` subschema.
  """
  def strip_constraints_recursive(schema) when is_map(schema) do
    schema
    |> Map.drop(["minimum", "maximum", "minLength", "maxLength", "minItems", "maxItems"])
    |> Map.new(fn
      {"properties", props} when is_map(props) ->
        {"properties", Map.new(props, fn {k, v} -> {k, strip_constraints_recursive(v)} end)}

      {"items", items} when is_map(items) ->
        {"items", strip_constraints_recursive(items)}

      {k, v} when is_map(v) ->
        {k, strip_constraints_recursive(v)}

      {k, v} ->
        {k, v}
    end)
  end

  def strip_constraints_recursive(value), do: value

  @doc """
  Apply the object-level requirements strict decoding mandates: every property
  is marked required and `additionalProperties` is set to `false`.
  """
  def enforce_strict_schema_requirements(
        %{"type" => "object", "properties" => properties} = schema
      ) do
    schema
    |> Map.put("required", Map.keys(properties))
    |> Map.put("additionalProperties", false)
  end

  def enforce_strict_schema_requirements(schema), do: schema
end
