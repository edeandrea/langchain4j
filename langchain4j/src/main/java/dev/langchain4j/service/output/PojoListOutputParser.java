package dev.langchain4j.service.output;

import java.util.ArrayList;
import java.util.List;
import java.util.function.Supplier;

class PojoListOutputParser<T> extends PojoCollectionOutputParser<T, List<T>> {

    PojoListOutputParser(Class<T> type) {
        super(type);
    }

    @Override
    Supplier<List<T>> emptyCollectionSupplier() {
        return ArrayList::new;
    }

    @Override
    Class<?> collectionType() {
        return List.class;
    }
}
