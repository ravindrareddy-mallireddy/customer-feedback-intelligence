import streamlit as st
import pandas as pd


def render(cfg, indexer, searcher, reranker):
    st.title("🔍 Semantic Search")
    st.markdown("Search 18,188 reviews using natural language. Results are reranked by relevance.")

    # Search input
    query = st.text_input(
        "Enter your search query",
        placeholder="e.g. battery life is terrible, fast delivery, easy to setup...",
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        top_k = st.slider("Number of results", 3, 20, 10)
    with col2:
        sentiment_filter = st.selectbox("Filter by sentiment", ["All", "positive", "negative"])
    with col3:
        category_filter = st.selectbox("Filter by category", [
            "All", "Electronics", "Clothing_Shoes_and_Jewelry",
            "Home_and_Kitchen", "Sports_and_Outdoors", "Books"
        ])

    if searcher is None:
        st.warning("Search index not available. Using embedding-only search.")

    if st.button("Search", type="primary") and query:
        with st.spinner("Searching..."):
            # Encode query
            query_emb = indexer.encode([query])[0]

            # Build ChromaDB filter
            where = None
            filters = {}
            if sentiment_filter != "All":
                filters["sentiment"] = sentiment_filter
            if category_filter != "All":
                filters["category"] = category_filter
            if filters:
                if len(filters) == 1:
                    where = filters
                else:
                    where = {"$and": [{k: v} for k, v in filters.items()]}

            # Search
            if where:
                results = searcher.search_chroma(query_emb, top_k=50, where=where)
            else:
                results = searcher.search_faiss(query_emb, top_k=50)

            # Rerank
            reranked = reranker.rerank(query, results, top_k=top_k)

            # Deduplicate
            seen_texts = set()
            unique_results = []
            for r in reranked:
                text = r["text"][:100]
                if text not in seen_texts:
                    seen_texts.add(text)
                    unique_results.append(r)

        st.markdown(f"**{len(unique_results)} results** for: *{query}*")
        st.markdown("---")

        for i, result in enumerate(unique_results, 1):
            score = result.get("rerank_score", 0)
            sentiment = result.get("sentiment", "unknown")
            category = result.get("category", "unknown")

            sentiment_color = {"positive": "🟢", "negative": "🔴", "neutral": "🟡"}.get(sentiment, "⚪")

            with st.expander(f"Result {i} — {sentiment_color} {sentiment.title()} | {category} | Score: {score:.3f}"):
                st.write(result["text"])
                col1, col2 = st.columns(2)
                with col1:
                    st.caption(f"Sentiment: {sentiment}")
                with col2:
                    st.caption(f"Category: {category}")

    elif not query:
        st.info("Enter a query above to search reviews.")

        # Show example queries
        st.markdown("**Example queries:**")
        examples = [
            "battery life is terrible",
            "great value for money",
            "fast delivery but poor packaging",
            "customer service was unhelpful",
            "easy to use and setup",
        ]
        for ex in examples:
            if st.button(ex, key=ex):
                st.session_state["query"] = ex
