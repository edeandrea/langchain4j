package dev.langchain4j.guardrail;

import static dev.langchain4j.internal.ValidationUtils.ensureNotNull;

import dev.langchain4j.data.message.AiMessage;
import dev.langchain4j.model.output.Response;
import java.util.Collections;
import java.util.List;
import java.util.Objects;
import java.util.Optional;
import java.util.stream.Collectors;
import java.util.stream.Stream;
import org.jspecify.annotations.NonNull;
import org.jspecify.annotations.Nullable;

/**
 * The result of the validation of an {@link OutputGuardrail}
 *
 * @param result
 *            The result of the output guardrail validation.
 * @param successfulText
 *            The successful text
 * @param successfulResult
 *            The successful result
 * @param failures
 *            The list of failures, empty if the validation succeeded.
 */
public record OutputGuardrailResult(
        Result result, @Nullable String successfulText, @Nullable Object successfulResult, List<Failure> failures)
        implements GuardrailResult<OutputGuardrailResult> {

    private static final OutputGuardrailResult SUCCESS = new OutputGuardrailResult();

    public OutputGuardrailResult {
        ensureNotNull(result, "result");
        failures = Optional.ofNullable(failures).orElseGet(List::of);
    }

    private OutputGuardrailResult() {
        this(Result.SUCCESS, null, null, Collections.emptyList());
    }

    private OutputGuardrailResult(@Nullable String successfulText) {
        this(Result.SUCCESS_WITH_RESULT, successfulText, null, Collections.emptyList());
    }

    private OutputGuardrailResult(@Nullable String successfulText, @Nullable Object successfulResult) {
        this(Result.SUCCESS_WITH_RESULT, successfulText, successfulResult, Collections.emptyList());
    }

    OutputGuardrailResult(@Nullable List<@NonNull Failure> failures, boolean fatal) {
        this(fatal ? Result.FATAL : Result.FAILURE, null, null, failures);
    }

    OutputGuardrailResult(Failure failure, boolean fatal) {
        // Using Stream.of().collect() here because we need a mutable list
        this(Stream.of(failure).collect(Collectors.toList()), fatal);
    }

    /**
     * Gets a successful output guardrail result
     */
    public static OutputGuardrailResult success() {
        return SUCCESS;
    }

    /**
     * Produces a successful result with specific success text
     *
     * @return The result of a successful output guardrail validation with a specific text.
     *
     * @param successfulText
     *            The text of the successful result.
     */
    public static OutputGuardrailResult successWith(@Nullable String successfulText) {
        return (successfulText == null) ? success() : new OutputGuardrailResult(successfulText);
    }

    /**
     * Produces a non-fatal failure
     *
     * @param successfulText
     *            The text of the successful result.
     * @param successfulResult
     *            The object generated by this successful result.
     * @return The result of a successful output guardrail validation with a specific text.
     */
    public static OutputGuardrailResult successWith(
            @Nullable String successfulText, @Nullable Object successfulResult) {
        return new OutputGuardrailResult(successfulText, successfulResult);
    }

    /**
     * Produces a non-fatal failure
     *
     * @param failures A list of {@link Failure}s
     *
     * @return The result of a failed output guardrail validation.
     */
    public static OutputGuardrailResult failure(@Nullable List<@NonNull Failure> failures) {
        return new OutputGuardrailResult(failures, false);
    }

    /**
     * Whether or not the guardrail is forcing a retry
     */
    public boolean isRetry() {
        return !isSuccess() && this.failures.stream().anyMatch(Failure::retry);
    }

    /**
     * Whether or not the guardrail is forcing a reprompt
     */
    public boolean isReprompt() {
        return !isSuccess()
                && this.failures.stream()
                                .map(Failure::reprompt)
                                .filter(Objects::nonNull)
                                .count()
                        > 0;
    }

    /**
     * Block all retries for this result
     */
    public OutputGuardrailResult blockRetry() {
        this.failures.set(0, this.failures.get(0).blockRetry());
        return this;
    }

    /**
     * Gets the reprompt message
     */
    public Optional<String> getReprompt() {
        return !isSuccess()
                ? this.failures.stream()
                        .map(Failure::reprompt)
                        .filter(Objects::nonNull)
                        .findFirst()
                : Optional.empty();
    }

    @Override
    public String toString() {
        return asString();
    }

    /**
     * Gets the response computed from the combination of the original {@link Response} in the {@link OutputGuardrailRequest}
     * and this result
     * @param params The output guardrail params
     * @param <T> The type of response
     * @return A response computed from the combination of the original {@link Response} in the {@link OutputGuardrailRequest}
     * and this result
     */
    public <T> T response(OutputGuardrailRequest params) {
        return (T) Optional.ofNullable(successfulResult()).orElseGet(() -> createResponse(params));
    }

    private Response<AiMessage> createResponse(OutputGuardrailRequest params) {
        var response = params.responseFromLLM();
        var aiMessage = response.aiMessage();
        var newAiMessage = aiMessage;

        if (hasRewrittenResult()) {
            newAiMessage = aiMessage.hasToolExecutionRequests()
                    ? AiMessage.from(successfulText(), aiMessage.toolExecutionRequests())
                    : AiMessage.from(successfulText());
        }

        return Response.from(
                newAiMessage,
                response.metadata().tokenUsage(),
                response.metadata().finishReason(),
                params.commonParams().variables());
    }

    /**
     * Represents an output guardrail failure
     *
     * @param message
     *            The failure message
     * @param cause
     *            The cause of the failure
     * @param guardrailClass
     *            The class that produced the failure
     * @param retry
     *            Whether the failure is intended to force a retry
     * @param reprompt
     *            The reprompt
     */
    public record Failure(
            String message,
            @Nullable Throwable cause,
            @Nullable Class<? extends Guardrail> guardrailClass,
            boolean retry,
            @Nullable String reprompt)
            implements GuardrailResult.Failure {

        public Failure {
            ensureNotNull(message, "message");
        }

        public Failure(String message) {
            this(message, null);
        }

        public Failure(String message, @Nullable Throwable cause) {
            this(message, cause, false);
        }

        public Failure(String message, @Nullable Throwable cause, boolean retry) {
            this(message, cause, null, retry, null);
        }

        public Failure(String message, @Nullable Throwable cause, boolean retry, @Nullable String reprompt) {
            this(message, cause, null, retry, reprompt);
        }

        @Override
        public Failure withGuardrailClass(Class<? extends Guardrail> guardrailClass) {
            ensureNotNull(guardrailClass, "guardrailClass");
            return new Failure(message(), cause(), guardrailClass, this.retry, this.reprompt);
        }

        /**
         * Create a failure from this failure that blocks retries
         */
        public Failure blockRetry() {
            return this.retry
                    ? new Failure(
                            "Retry or reprompt is not allowed after a rewritten output",
                            cause(),
                            this.guardrailClass,
                            false,
                            this.reprompt)
                    : this;
        }

        @Override
        public String toString() {
            return asString();
        }
    }
}
