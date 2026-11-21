package dev.langchain4j.guardrails;

import dev.langchain4j.Experimental;
import dev.langchain4j.agent.tool.ToolSpecification;
import dev.langchain4j.guardrail.ToolInputGuardrail;
import dev.langchain4j.guardrail.ToolInputGuardrailRequest;
import dev.langchain4j.guardrail.ToolInputGuardrailResult;
import dev.langchain4j.model.chat.request.json.JsonAnyOfSchema;
import dev.langchain4j.model.chat.request.json.JsonArraySchema;
import dev.langchain4j.model.chat.request.json.JsonBooleanSchema;
import dev.langchain4j.model.chat.request.json.JsonEnumSchema;
import dev.langchain4j.model.chat.request.json.JsonIntegerSchema;
import dev.langchain4j.model.chat.request.json.JsonNullSchema;
import dev.langchain4j.model.chat.request.json.JsonNumberSchema;
import dev.langchain4j.model.chat.request.json.JsonObjectSchema;
import dev.langchain4j.model.chat.request.json.JsonReferenceSchema;
import dev.langchain4j.model.chat.request.json.JsonSchemaElement;
import dev.langchain4j.model.chat.request.json.JsonStringSchema;
import java.math.BigDecimal;
import java.math.BigInteger;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

/**
 * A {@link ToolInputGuardrail} that checks the arguments chosen by the LLM against the JSON schema of the tool's
 * parameters ({@link ToolSpecification#parameters()}) before the tool runs.
 * <p>
 *     It reports, at any depth:
 * </p>
 * <ul>
 *     <li>a required property that is missing, or {@code null} when its schema does not allow {@code null}</li>
 *     <li>a value of the wrong JSON type: e.g. a string or {@code 2.7} for an integer, a number for a string</li>
 *     <li>a value that is not one of the values of an enum</li>
 *     <li>a property that is not declared, when the object schema sets {@code additionalProperties} to {@code false}</li>
 * </ul>
 * <p>
 *     Without it, these values are converted when the arguments are bound to the tool method: inside an object
 *     parameter, a missing or {@code null} {@code int} becomes {@code 0} and {@code 2.7} becomes {@code 2}, and a
 *     missing object parameter becomes {@code null}.
 * </p>
 * <p>
 *     A violation is a non-fatal failure: the tool does not run, and the LLM gets an error tool result that lists
 *     each violation by its path and the expected type, without the values it sent, so it can call the tool again.
 * </p>
 * <p>
 *     The check is strict JSON schema typing: a string such as {@code "42"} does not match an integer, and an enum
 *     value must match exactly. An integral number written with a fraction, such as {@code 3.0}, matches an
 *     integer. A {@code null} value for a property that is not required is accepted. {@link JsonReferenceSchema}s
 *     are resolved against the definitions of the root object schema; raw schemas and references that cannot be
 *     resolved are not checked. Tools without a {@link ToolSpecification} or without parameters are not checked.
 *     Arguments that are not a JSON object are left to the {@code ToolArgumentsErrorHandler} of the AI service,
 *     as for any tool input guardrail.
 * </p>
 * <p>
 *     The guardrail has no state, so one instance can be shared by every tool and every invocation:
 * </p>
 * <pre>{@code
 * AiServices.builder(Assistant.class)
 *         .chatModel(chatModel)
 *         .tools(new OrderTools())
 *         .toolInputGuardrails(new JsonSchemaToolInputGuardrail())
 *         .build();
 * }</pre>
 *
 * @since 1.25.0
 */
@Experimental
public class JsonSchemaToolInputGuardrail implements ToolInputGuardrail {

    /**
     * The maximum number of violations listed in the failure message.
     */
    static final int MAX_LISTED_VIOLATIONS = 20;

    @Override
    public ToolInputGuardrailResult validate(ToolInputGuardrailRequest request) {
        ToolSpecification toolSpecification = request.toolSpecification();
        if (toolSpecification == null || toolSpecification.parameters() == null) {
            return success();
        }
        JsonObjectSchema parameters = toolSpecification.parameters();

        // arguments that are not a JSON object throw a ToolArgumentsException here, which goes to the
        // ToolArgumentsErrorHandler of the AI service like any other arguments error
        Map<String, Object> arguments = request.argumentsAsMap();

        List<String> violations = new ArrayList<>();
        new Checker(parameters.definitions()).checkObject(arguments, parameters, "", violations);

        return violations.isEmpty() ? success() : failure(violationsMessage(request.toolName(), violations));
    }

    /**
     * Builds the message sent to the LLM when the arguments do not match the schema.
     * Override it to change the wording.
     *
     * @param toolName the name of the tool
     * @param violations the violations, each as {@code "<path>: <problem>"}; they do not contain argument values
     * @return the message sent to the LLM
     */
    protected String violationsMessage(String toolName, List<String> violations) {
        StringBuilder message =
                new StringBuilder("The arguments of tool \"%s\" do not match its parameters:".formatted(toolName));
        violations.stream()
                .limit(MAX_LISTED_VIOLATIONS)
                .forEach(v -> message.append("\n- ").append(v));
        if (violations.size() > MAX_LISTED_VIOLATIONS) {
            message.append("\n- and %d more".formatted(violations.size() - MAX_LISTED_VIOLATIONS));
        }
        return message.append("\nCall the tool again with corrected arguments.").toString();
    }

    private record Checker(Map<String, JsonSchemaElement> definitions) {

        void check(Object value, JsonSchemaElement schema, String path, List<String> violations) {
            if (schema instanceof JsonReferenceSchema reference) {
                JsonSchemaElement resolved = definitions == null ? null : definitions.get(reference.reference());
                if (resolved != null) {
                    check(value, resolved, path, violations);
                }
            } else if (schema instanceof JsonAnyOfSchema anyOf) {
                checkAnyOf(value, anyOf, path, violations);
            } else if (schema instanceof JsonNullSchema) {
                if (value != null) {
                    violations.add(at(path) + "expected null");
                }
            } else if (value == null) {
                if (isChecked(schema)) {
                    violations.add(at(path) + "expected " + expected(schema) + ", got null");
                }
            } else if (schema instanceof JsonObjectSchema object) {
                if (value instanceof Map<?, ?> map) {
                    checkObject(map, object, path, violations);
                } else {
                    violations.add(at(path) + "expected an object");
                }
            } else if (schema instanceof JsonArraySchema array) {
                if (value instanceof List<?> list) {
                    if (array.items() != null) {
                        for (int i = 0; i < list.size(); i++) {
                            check(list.get(i), array.items(), path + "[" + i + "]", violations);
                        }
                    }
                } else {
                    violations.add(at(path) + "expected an array");
                }
            } else if (schema instanceof JsonEnumSchema enumSchema) {
                if (!(value instanceof String string)
                        || !enumSchema.enumValues().contains(string)) {
                    violations.add(at(path) + "expected one of " + enumSchema.enumValues());
                }
            } else if (schema instanceof JsonStringSchema) {
                if (!(value instanceof String)) {
                    violations.add(at(path) + "expected a string");
                }
            } else if (schema instanceof JsonIntegerSchema) {
                if (!isInteger(value)) {
                    violations.add(at(path) + "expected an integer");
                }
            } else if (schema instanceof JsonNumberSchema) {
                if (!(value instanceof Number)) {
                    violations.add(at(path) + "expected a number");
                }
            } else if (schema instanceof JsonBooleanSchema) {
                if (!(value instanceof Boolean)) {
                    violations.add(at(path) + "expected a boolean");
                }
            }
            // JsonRawSchema and unknown elements are not checked
        }

        void checkObject(Map<?, ?> object, JsonObjectSchema schema, String path, List<String> violations) {
            Map<String, JsonSchemaElement> properties = schema.properties() == null ? Map.of() : schema.properties();

            if (schema.required() != null) {
                for (String required : schema.required()) {
                    if (!object.containsKey(required)) {
                        violations.add(at(child(path, required)) + "required but missing");
                    } else if (object.get(required) == null && properties.containsKey(required)) {
                        check(null, properties.get(required), child(path, required), violations);
                    }
                }
            }

            for (Map.Entry<?, ?> entry : object.entrySet()) {
                String name = String.valueOf(entry.getKey());
                JsonSchemaElement propertySchema = properties.get(name);
                if (propertySchema == null) {
                    if (Boolean.FALSE.equals(schema.additionalProperties())) {
                        violations.add(at(child(path, name)) + "not a declared parameter");
                    }
                } else if (entry.getValue() != null) {
                    check(entry.getValue(), propertySchema, child(path, name), violations);
                }
                // a null value of a required property was checked above; of an optional one, it is accepted
            }
        }

        private void checkAnyOf(Object value, JsonAnyOfSchema anyOf, String path, List<String> violations) {
            if (anyOf.anyOf() == null || anyOf.anyOf().isEmpty()) {
                return;
            }
            for (JsonSchemaElement option : anyOf.anyOf()) {
                List<String> optionViolations = new ArrayList<>();
                check(value, option, path, optionViolations);
                if (optionViolations.isEmpty()) {
                    return;
                }
            }
            violations.add(at(path) + "does not match any of the allowed schemas");
        }

        private boolean isChecked(JsonSchemaElement schema) {
            return schema instanceof JsonObjectSchema
                    || schema instanceof JsonArraySchema
                    || schema instanceof JsonEnumSchema
                    || schema instanceof JsonStringSchema
                    || schema instanceof JsonIntegerSchema
                    || schema instanceof JsonNumberSchema
                    || schema instanceof JsonBooleanSchema;
        }

        private static String expected(JsonSchemaElement schema) {
            if (schema instanceof JsonObjectSchema) return "an object";
            if (schema instanceof JsonArraySchema) return "an array";
            if (schema instanceof JsonEnumSchema enumSchema) return "one of " + enumSchema.enumValues();
            if (schema instanceof JsonStringSchema) return "a string";
            if (schema instanceof JsonIntegerSchema) return "an integer";
            if (schema instanceof JsonNumberSchema) return "a number";
            return "a boolean";
        }

        private static boolean isInteger(Object value) {
            if (value instanceof Integer
                    || value instanceof Long
                    || value instanceof Short
                    || value instanceof Byte
                    || value instanceof BigInteger) {
                return true;
            }
            if (value instanceof BigDecimal decimal) {
                return decimal.signum() == 0 || decimal.stripTrailingZeros().scale() <= 0;
            }
            if (value instanceof Double || value instanceof Float) {
                double d = ((Number) value).doubleValue();
                return Double.isFinite(d) && d == Math.rint(d);
            }
            return false;
        }

        private static String child(String path, String name) {
            return path.isEmpty() ? name : path + "." + name;
        }

        private static String at(String path) {
            return (path.isEmpty() ? "arguments" : path) + ": ";
        }
    }
}
