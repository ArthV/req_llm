defmodule ReqLLM.Providers.Anthropic.AdapterHelpersTest do
  use ExUnit.Case, async: true

  @moduletag category: :core
  @moduletag provider: :anthropic

  alias ReqLLM.Providers.Anthropic
  alias ReqLLM.Providers.Anthropic.AdapterHelpers

  @json_schema %{
    "type" => "object",
    "properties" => %{
      "question" => %{"type" => "string"},
      "options" => %{
        "type" => "array",
        "items" => %{"type" => "string"},
        "minItems" => 4,
        "maxItems" => 4
      }
    },
    "required" => ["options", "question"],
    "additionalProperties" => true
  }

  defp tool_format(extra_opts) do
    {:ok, compiled} = ReqLLM.Schema.compile(@json_schema)
    opts = Keyword.put(extra_opts, :compiled_schema, compiled)
    {_context, updated_opts} = AdapterHelpers.prepare_structured_output_context(%{}, opts)
    [tool | _] = Keyword.fetch!(updated_opts, :tools)
    Anthropic.tool_to_anthropic_format(tool)
  end

  describe "prepare_structured_output_context/2 default (best-effort)" do
    test "does not mark the tool strict and leaves the schema untouched" do
      formatted = tool_format([])

      refute Map.has_key?(formatted, :strict)
      assert formatted[:input_schema]["properties"]["options"]["minItems"] == 4
      assert formatted[:input_schema]["properties"]["options"]["maxItems"] == 4
      assert formatted[:input_schema]["additionalProperties"] == true
    end

    test "auto mode is treated as best-effort" do
      formatted = tool_format(anthropic_structured_output_mode: :auto)

      refute Map.has_key?(formatted, :strict)
    end
  end

  describe "prepare_structured_output_context/2 strict mode" do
    test "tool_strict marks the tool strict and reduces the schema to the supported subset" do
      formatted = tool_format(anthropic_structured_output_mode: :tool_strict)
      options = formatted[:input_schema]["properties"]["options"]

      assert formatted[:strict] == true
      refute Map.has_key?(options, "minItems")
      refute Map.has_key?(options, "maxItems")
      assert options["type"] == "array"
      assert options["items"] == %{"type" => "string"}
      assert formatted[:input_schema]["additionalProperties"] == false
      assert Enum.sort(formatted[:input_schema]["required"]) == ["options", "question"]
    end

    test "mode resolves when nested under :provider_options" do
      formatted = tool_format(provider_options: [anthropic_structured_output_mode: :tool_strict])

      assert formatted[:strict] == true
      refute Map.has_key?(formatted[:input_schema]["properties"]["options"], "maxItems")
    end

    test "json_schema mode is also strict" do
      formatted = tool_format(anthropic_structured_output_mode: :json_schema)

      assert formatted[:strict] == true
    end
  end
end
