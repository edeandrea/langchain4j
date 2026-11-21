package dev.langchain4j.guardrails;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import dev.langchain4j.agent.tool.P;
import dev.langchain4j.agent.tool.Tool;
import dev.langchain4j.agent.tool.ToolExecutionRequest;
import dev.langchain4j.agent.tool.ToolSpecification;
import dev.langchain4j.agent.tool.ToolSpecifications;
import dev.langchain4j.data.message.ToolExecutionResultMessage;
import dev.langchain4j.exception.ToolArgumentsException;
import dev.langchain4j.guardrail.GuardrailRequestParams;
import dev.langchain4j.guardrail.ToolInputGuardrailRequest;
import dev.langchain4j.guardrail.ToolInputGuardrailResult;
import dev.langchain4j.invocation.InvocationContext;
import dev.langchain4j.model.chat.request.json.JsonAnyOfSchema;
import dev.langchain4j.model.chat.request.json.JsonArraySchema;
import dev.langchain4j.model.chat.request.json.JsonIntegerSchema;
import dev.langchain4j.model.chat.request.json.JsonNullSchema;
import dev.langchain4j.model.chat.request.json.JsonObjectSchema;
import dev.langchain4j.model.chat.request.json.JsonReferenceSchema;
import dev.langchain4j.model.chat.request.json.JsonStringSchema;
import dev.langchain4j.observability.api.AiServiceListenerRegistrar;
import dev.langchain4j.service.AiServices;
import dev.langchain4j.test.mock.MockChatModel;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;
import java.util.stream.IntStream;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;

class JsonSchemaToolInputGuardrailTest {

    enum Priority {
        LOW,
        HIGH
    }

    record Order(String item, int quantity, double unitPrice, boolean express, Priority priority, List<String> tags) {}

    static class OrderTools {

        final List<Order> orders = new ArrayList<>();
        final List<Integer> quantities = new ArrayList<>();

        @Tool("Places an order")
        String placeOrder(Order order) {
            orders.add(order);
            return "placed";
        }

        @Tool("Sets a quantity")
        String setQuantity(@P("the quantity") int quantity) {
            quantities.add(quantity);
            return "set";
        }
    }

    interface Assistant {
        String chat(String message);
    }

    private static final ToolSpecification PLACE_ORDER = spec("placeOrder");
    private static final ToolSpecification SET_QUANTITY = spec("setQuantity");

    private final JsonSchemaToolInputGuardrail guardrail = new JsonSchemaToolInputGuardrail();

    // ---------- valid arguments ----------

    @Test
    void should_accept_valid_nested_arguments() {
        assertSuccess(PLACE_ORDER, """
                {"order": {"item": "pen", "quantity": 3, "unitPrice": 1.5, "express": true,
                           "priority": "HIGH", "tags": ["a", "b"]}}
                """);
    }

    @ParameterizedTest
    @ValueSource(strings = {"3", "3.0", "-7", "0", "9223372036854775807", "12345678901234567890"})
    void should_accept_integral_numbers_for_an_integer(String quantity) {
        assertSuccess(SET_QUANTITY, "{\"quantity\": " + quantity + "}");
    }

    @Test
    void should_accept_an_integer_for_a_number() {
        assertSuccess(PLACE_ORDER, """
                {"order": {"item": "pen", "quantity": 3, "unitPrice": 2, "express": false,
                           "priority": "LOW", "tags": []}}
                """);
    }

    @Test
    void should_skip_tools_without_specification_or_parameters() {
        assertThat(guardrail.validate(request(null, "{\"anything\": 1}")).isSuccess())
                .isTrue();
        ToolSpecification noParameters =
                ToolSpecification.builder().name("ping").build();
        assertThat(guardrail.validate(request(noParameters, "{}")).isSuccess()).isTrue();
    }

    // ---------- invalid arguments ----------

    @Test
    void should_report_a_fraction_for_an_integer_at_top_level_and_nested() {
        assertViolations(SET_QUANTITY, "{\"quantity\": 2.7}", "quantity: expected an integer");
        assertViolations(PLACE_ORDER, """
                {"order": {"item": "pen", "quantity": 2.7, "unitPrice": 1.5, "express": true,
                           "priority": "LOW", "tags": []}}
                """, "order.quantity: expected an integer");
    }

    @Test
    void should_report_missing_required_properties_at_any_depth() {
        assertViolations(SET_QUANTITY, "{}", "quantity: required but missing");
        assertViolations(PLACE_ORDER, "{}", "order: required but missing");
        assertViolations(
                PLACE_ORDER,
                "{\"order\": {}}",
                "order.item: required but missing",
                "order.quantity: required but missing",
                "order.unitPrice: required but missing",
                "order.express: required but missing",
                "order.priority: required but missing",
                "order.tags: required but missing");
    }

    @Test
    void should_report_null_for_a_required_property() {
        assertViolations(SET_QUANTITY, "{\"quantity\": null}", "quantity: expected an integer, got null");
        assertViolations(PLACE_ORDER, "{\"order\": null}", "order: expected an object, got null");
    }

    @Test
    void should_report_wrong_json_types() {
        assertViolations(
                PLACE_ORDER,
                """
                {"order": {"item": 42, "quantity": "3", "unitPrice": "1.5", "express": "true",
                           "priority": "HIGH", "tags": "a"}}
                """,
                "order.item: expected a string",
                "order.quantity: expected an integer",
                "order.unitPrice: expected a number",
                "order.express: expected a boolean",
                "order.tags: expected an array");
        assertViolations(PLACE_ORDER, "{\"order\": [1, 2]}", "order: expected an object");
    }

    @Test
    void should_report_array_items_by_index() {
        assertViolations(
                PLACE_ORDER, """
                {"order": {"item": "pen", "quantity": 3, "unitPrice": 1.5, "express": true,
                           "priority": "HIGH", "tags": ["a", 7, null]}}
                """, "order.tags[1]: expected a string", "order.tags[2]: expected a string, got null");
    }

    @Test
    void should_report_a_value_outside_an_enum() {
        assertViolations(PLACE_ORDER, """
                {"order": {"item": "pen", "quantity": 3, "unitPrice": 1.5, "express": true,
                           "priority": "URGENT", "tags": []}}
                """, "order.priority: expected one of [LOW, HIGH]");
    }

    @Test
    void should_report_a_numeric_string_for_an_integer() {
        assertViolations(SET_QUANTITY, "{\"quantity\": \"NaN\"}", "quantity: expected an integer");
    }

    @ParameterizedTest
    @ValueSource(strings = {"{not json", "[1, 2]", "42", "\"text\""})
    void should_leave_arguments_that_are_not_a_json_object_to_the_arguments_error_handler(String arguments) {
        assertThatThrownBy(() -> guardrail.validate(request(SET_QUANTITY, arguments)))
                .isInstanceOf(ToolArgumentsException.class);
    }

    @Test
    void should_not_put_argument_values_in_the_message() {
        ToolInputGuardrailResult result = guardrail.validate(request(PLACE_ORDER, """
                {"order": {"item": 4242, "quantity": "SECRET-VALUE", "unitPrice": 1.5, "express": true,
                           "priority": "TOP-SECRET", "tags": []}}
                """));
        assertThat(result.failureMessage()).doesNotContain("4242", "SECRET-VALUE", "TOP-SECRET");
    }

    @Test
    void should_cap_the_listed_violations() {
        JsonObjectSchema.Builder parameters = JsonObjectSchema.builder();
        List<String> names = IntStream.range(0, 25).mapToObj(i -> "p" + i).toList();
        names.forEach(parameters::addIntegerProperty);
        parameters.required(names);
        ToolSpecification wide = ToolSpecification.builder()
                .name("wide")
                .parameters(parameters.build())
                .build();

        ToolInputGuardrailResult result = guardrail.validate(request(wide, "{}"));

        assertThat(result.failureMessage()).contains("p19: required but missing", "and 5 more");
        assertThat(result.failureMessage()).doesNotContain("p20: required but missing");
    }

    // ---------- schema features ----------

    @Test
    void should_reject_undeclared_properties_only_when_additional_properties_is_false() {
        JsonObjectSchema strict = JsonObjectSchema.builder()
                .addStringProperty("name")
                .additionalProperties(false)
                .build();
        JsonObjectSchema open =
                JsonObjectSchema.builder().addStringProperty("name").build();

        assertViolations(
                ToolSpecification.builder().name("strict").parameters(strict).build(),
                "{\"name\": \"a\", \"extra\": 1}",
                "extra: not a declared parameter");
        assertSuccess(
                ToolSpecification.builder().name("open").parameters(open).build(), "{\"name\": \"a\", \"extra\": 1}");
    }

    @Test
    void should_accept_null_for_an_optional_property_and_for_a_nullable_required_one() {
        JsonObjectSchema parameters = JsonObjectSchema.builder()
                .addStringProperty("optional")
                .addProperty(
                        "nullable",
                        JsonAnyOfSchema.builder()
                                .anyOf(new JsonStringSchema(), new JsonNullSchema())
                                .build())
                .required("nullable")
                .build();
        ToolSpecification tool =
                ToolSpecification.builder().name("t").parameters(parameters).build();

        assertSuccess(tool, "{\"optional\": null, \"nullable\": null}");
        assertViolations(tool, "{\"nullable\": 5}", "nullable: does not match any of the allowed schemas");
    }

    @Test
    void should_resolve_references_against_root_definitions() {
        JsonObjectSchema node = JsonObjectSchema.builder()
                .addIntegerProperty("value")
                .addProperty(
                        "next", JsonReferenceSchema.builder().reference("node").build())
                .required("value")
                .build();
        JsonObjectSchema parameters = JsonObjectSchema.builder()
                .addProperty(
                        "head", JsonReferenceSchema.builder().reference("node").build())
                .required("head")
                .definitions(Map.of("node", node))
                .build();
        ToolSpecification tool =
                ToolSpecification.builder().name("list").parameters(parameters).build();

        assertSuccess(tool, "{\"head\": {\"value\": 1, \"next\": {\"value\": 2}}}");
        assertViolations(
                tool, "{\"head\": {\"value\": 1, \"next\": {\"value\": 2.5}}}", "head.next.value: expected an integer");
    }

    @Test
    void should_not_check_unresolvable_references() {
        JsonObjectSchema parameters = JsonObjectSchema.builder()
                .addProperty(
                        "x", JsonReferenceSchema.builder().reference("missing").build())
                .build();
        assertSuccess(
                ToolSpecification.builder().name("t").parameters(parameters).build(), "{\"x\": 1}");
    }

    @Test
    void should_check_nested_arrays() {
        JsonObjectSchema parameters = JsonObjectSchema.builder()
                .addProperty(
                        "matrix",
                        JsonArraySchema.builder()
                                .items(JsonArraySchema.builder()
                                        .items(new JsonIntegerSchema())
                                        .build())
                                .build())
                .build();
        ToolSpecification tool =
                ToolSpecification.builder().name("m").parameters(parameters).build();

        assertSuccess(tool, "{\"matrix\": [[1, 2], [3]]}");
        assertViolations(tool, "{\"matrix\": [[1, 2], [3, 4.5]]}", "matrix[1][1]: expected an integer");
    }

    // ---------- through an AI service ----------

    @Test
    void should_stop_a_nested_fraction_and_let_the_llm_retry() {
        OrderTools tools = new OrderTools();
        MockChatModel model = MockChatModel.thatCallsTool("placeOrder", """
                        {"order": {"item": "pen", "quantity": 2.7, "unitPrice": 1.5, "express": true,
                                   "priority": "LOW", "tags": []}}
                        """)
                .thenCallsTool("placeOrder", """
                        {"order": {"item": "pen", "quantity": 3, "unitPrice": 1.5, "express": true,
                                   "priority": "LOW", "tags": []}}
                        """)
                .thenResponds("done");

        Assistant assistant = AiServices.builder(Assistant.class)
                .chatModel(model)
                .tools(tools)
                .toolInputGuardrails(guardrail)
                .build();

        assertThat(assistant.chat("order 3 pens")).isEqualTo("done");

        // without the guardrail, the first call runs with quantity = 2
        assertThat(tools.orders).extracting(Order::quantity).containsExactly(3);
        ToolExecutionResultMessage firstResult = toolResults(model).get(0);
        assertThat(firstResult.isError()).isTrue();
        assertThat(firstResult.text()).contains("order.quantity: expected an integer");
    }

    @Test
    void should_stop_an_empty_nested_object() {
        OrderTools tools = new OrderTools();
        MockChatModel model =
                MockChatModel.thatCallsTool("placeOrder", "{\"order\": {}}").thenResponds("done");

        Assistant assistant = AiServices.builder(Assistant.class)
                .chatModel(model)
                .tools(tools)
                .toolInputGuardrails(guardrail)
                .build();

        assistant.chat("order something");

        // without the guardrail, the tool runs with Order[item=null, quantity=0, unitPrice=0.0, express=false, ...]
        assertThat(tools.orders).isEmpty();
        assertThat(toolResults(model).get(0).text()).contains("order.quantity: required but missing");
    }

    // ---------- helpers ----------

    private static ToolSpecification spec(String name) {
        return ToolSpecifications.toolSpecificationsFrom(OrderTools.class).stream()
                .filter(s -> s.name().equals(name))
                .findFirst()
                .orElseThrow();
    }

    private static List<ToolExecutionResultMessage> toolResults(MockChatModel model) {
        return model.lastRequest().messages().stream()
                .filter(ToolExecutionResultMessage.class::isInstance)
                .map(ToolExecutionResultMessage.class::cast)
                .collect(Collectors.toList());
    }

    private static ToolInputGuardrailRequest request(ToolSpecification spec, String arguments) {
        return ToolInputGuardrailRequest.builder()
                .toolExecutionRequest(ToolExecutionRequest.builder()
                        .id("1")
                        .name(spec == null ? "tool" : spec.name())
                        .arguments(arguments)
                        .build())
                .toolSpecification(spec)
                .requestParams(GuardrailRequestParams.builder()
                        .userMessageTemplate("t")
                        .variables(Map.of())
                        .invocationContext(InvocationContext.builder().build())
                        .aiServiceListenerRegistrar(AiServiceListenerRegistrar.newInstance())
                        .build())
                .build();
    }

    private void assertSuccess(ToolSpecification spec, String arguments) {
        ToolInputGuardrailResult result = guardrail.validate(request(spec, arguments));
        assertThat(result.isSuccess()).as(result.failureMessage()).isTrue();
    }

    private void assertViolations(ToolSpecification spec, String arguments, String... expected) {
        ToolInputGuardrailResult result = guardrail.validate(request(spec, arguments));
        assertThat(result.isSuccess()).isFalse();
        assertThat(result.isFatal()).isFalse();
        String message = result.failureMessage();
        List<String> listed = message.lines()
                .filter(l -> l.startsWith("- "))
                .map(l -> l.substring(2))
                .toList();
        assertThat(listed).containsExactly(expected);
    }
}
