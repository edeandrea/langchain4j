package dev.langchain4j.guardrail;

import dev.langchain4j.data.message.AiMessage;
import java.util.Arrays;
import org.jspecify.annotations.Nullable;

/**
 * An output guardrail is a rule that is applied to the output of the model to ensure that the output is safe and meets
 * the expectations.
 * <p>
 * In the case of reprompting, the reprompt message is added to the LLM context and the request is retried.
 * <p>
 * The maximum number of retries is configurable, defaulting to {@link dev.langchain4j.guardrail.config.OutputGuardrailsConfig#MAX_RETRIES_DEFAULT}.
 */
public interface OutputGuardrail extends Guardrail<OutputGuardrailParams, OutputGuardrailResult> {
    /**
     * Validates the response from the LLM.
     *
     * @param responseFromLLM
     *            the response from the LLM
     */
    default OutputGuardrailResult validate(AiMessage responseFromLLM) {
        return failure("Validation not implemented");
    }

    /**
     * Validates the response from the LLM.
     * <p>
     * Unlike {@link #validate(AiMessage)}, this method allows to access the memory and the augmentation result (in the
     * case of a RAG).
     * <p>
     * Implementation must not attempt to write to the memory or the augmentation result.
     *
     * @param params
     *            the parameters, including the response from the LLM, the memory, and the augmentation result.
     */
    @Override
    default OutputGuardrailResult validate(OutputGuardrailParams params) {
        return validate(params.responseFromLLM().aiMessage());
    }

    /**
     * Produces a successful result without any successful text
     *
     * @return The result of a successful output guardrail validation.
     */
    default OutputGuardrailResult success() {
        return OutputGuardrailResult.success();
    }

    /**
     * Produces a successful result with specific success text
     *
     * @return The result of a successful output guardrail validation with a specific text.
     *
     * @param successfulText
     *            The text of the successful result.
     */
    default OutputGuardrailResult successWith(@Nullable String successfulText) {
        return OutputGuardrailResult.successWith(successfulText);
    }

    /**
     * Produces a non-fatal failure
     *
     * @return The result of a successful output guardrail validation with a specific text.
     *
     * @param successfulText
     *            The text of the successful result.
     * @param successfulResult
     *            The object generated by this successful result.
     */
    default OutputGuardrailResult successWith(@Nullable String successfulText, @Nullable Object successfulResult) {
        return OutputGuardrailResult.successWith(successfulText, successfulResult);
    }

    /**
     * Produces a non-fatal failure
     *
     * @param message
     *            A message describing the failure.
     *
     * @return The result of a failed output guardrail validation.
     */
    default OutputGuardrailResult failure(String message) {
        return new OutputGuardrailResult(new OutputGuardrailResult.Failure(message), false);
    }

    /**
     * Produces a non-fatal failure
     *
     * @param message
     *            A message describing the failure.
     * @param cause
     *            The exception that caused this failure.
     *
     * @return The result of a failed output guardrail validation.
     */
    default OutputGuardrailResult failure(String message, @Nullable Throwable cause) {
        return new OutputGuardrailResult(new OutputGuardrailResult.Failure(message, cause), false);
    }

    /**
     * Produces a fatal failure
     *
     * @param message
     *            A message describing the failure.
     *
     * @return The result of a fatally failed output guardrail validation, blocking the evaluation of any other
     *         subsequent validation.
     */
    default OutputGuardrailResult fatal(String message) {
        return new OutputGuardrailResult(Arrays.asList(new OutputGuardrailResult.Failure(message)), true);
    }

    /**
     * Produces a fatal failure
     *
     * @param message
     *            A message describing the failure.
     * @param cause
     *            The exception that caused this failure.
     *
     * @return The result of a fatally failed output guardrail validation, blocking the evaluation of any other
     *         subsequent validation.
     */
    default OutputGuardrailResult fatal(String message, @Nullable Throwable cause) {
        return new OutputGuardrailResult(Arrays.asList(new OutputGuardrailResult.Failure(message, cause)), true);
    }

    /**
     * @param message
     *            A message describing the failure.
     *
     * @return The result of a fatally failed output guardrail validation, blocking the evaluation of any other
     *         subsequent validation and triggering a retry with the same user prompt.
     */
    default OutputGuardrailResult retry(String message) {
        return new OutputGuardrailResult(Arrays.asList(new OutputGuardrailResult.Failure(message, null, true)), true);
    }

    /**
     * @param message
     *            A message describing the failure.
     * @param cause
     *            The exception that caused this failure.
     *
     * @return The result of a fatally failed output guardrail validation, blocking the evaluation of any other
     *         subsequent validation and triggering a retry with the same user prompt.
     */
    default OutputGuardrailResult retry(String message, @Nullable Throwable cause) {
        return new OutputGuardrailResult(Arrays.asList(new OutputGuardrailResult.Failure(message, cause, true)), true);
    }

    /**
     * @param message
     *            A message describing the failure.
     * @param reprompt
     *            The new prompt to be used for the retry.
     *
     * @return The result of a fatally failed output guardrail validation, blocking the evaluation of any other
     *         subsequent validation and triggering a retry with a new user prompt.
     */
    default OutputGuardrailResult reprompt(String message, @Nullable String reprompt) {
        return new OutputGuardrailResult(
                Arrays.asList(new OutputGuardrailResult.Failure(message, null, true, reprompt)), true);
    }

    /**
     * @param message
     *            A message describing the failure.
     * @param cause
     *            The exception that caused this failure.
     * @param reprompt
     *            The new prompt to be used for the retry.
     *
     * @return The result of a fatally failed output guardrail validation, blocking the evaluation of any other
     *         subsequent validation and triggering a retry with a new user prompt.
     */
    default OutputGuardrailResult reprompt(String message, @Nullable Throwable cause, @Nullable String reprompt) {
        return new OutputGuardrailResult(
                Arrays.asList(new OutputGuardrailResult.Failure(message, cause, true, reprompt)), true);
    }
}
