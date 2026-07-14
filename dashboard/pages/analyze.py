import streamlit as st
import plotly.graph_objects as go


def render(cfg, sentiment_clf, aspect_clf):
    st.title("🏷️ Analyze a Review")
    st.markdown("Paste any product review to get instant sentiment and aspect analysis.")

    review_text = st.text_area(
        "Enter review text",
        height=150,
        placeholder="e.g. The battery life on this product is terrible. It barely lasts 2 hours and the customer service was unhelpful when I tried to get a replacement...",
    )

    if st.button("Analyze", type="primary") and review_text:
        with st.spinner("Analyzing..."):
            # Sentiment prediction
            label_map = cfg["sentiment"]["label_map"]
            id2label = {v: k for k, v in label_map.items()}
            sentiment_pred = sentiment_clf.predict([review_text])[0]
            sentiment_label = id2label.get(sentiment_pred, "unknown")

            # Aspect prediction
            aspects = cfg["models"]["aspect"]["aspects"]
            aspect_preds = aspect_clf.predict([review_text], threshold=0.5)[0]
            detected_aspects = [aspects[i] for i, v in enumerate(aspect_preds) if v == 1]

        # Display sentiment
        st.markdown("---")
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Sentiment")
            color = {"positive": "green", "negative": "red", "neutral": "orange"}.get(sentiment_label, "gray")
            icon = {"positive": "😊", "negative": "😞", "neutral": "😐"}.get(sentiment_label, "❓")
            st.markdown(f"## {icon} :{color}[{sentiment_label.upper()}]")

        with col2:
            st.subheader("Detected Aspects")
            if detected_aspects:
                for asp in detected_aspects:
                    asp_icon = {
                        "price": "💰",
                        "quality": "⭐",
                        "delivery": "🚚",
                        "customer_service": "🎧",
                        "packaging": "📦",
                        "usability": "🔧",
                    }.get(asp, "•")
                    st.markdown(f"{asp_icon} **{asp.replace('_', ' ').title()}**")
            else:
                st.info("No specific aspects detected.")

        # Aspect confidence bars
        st.markdown("---")
        st.subheader("Aspect Confidence")

        import torch
        import numpy as np
        max_len = cfg["data"]["max_text_length"]
        encodings = aspect_clf.tokenizer(
            review_text, max_length=max_len,
            padding=True, truncation=True, return_tensors="pt"
        )
        encodings = {k: v.to(aspect_clf.device) for k, v in encodings.items()}
        with torch.no_grad():
            outputs = aspect_clf.model(**encodings)
        probs = torch.sigmoid(outputs.logits).cpu().numpy()[0]

        fig = go.Figure(go.Bar(
            x=probs,
            y=[a.replace("_", " ").title() for a in aspects],
            orientation="h",
            marker_color=["#4CAF50" if p >= 0.5 else "#E0E0E0" for p in probs],
        ))
        fig.add_vline(x=0.5, line_dash="dash", line_color="red", annotation_text="Threshold")
        fig.update_layout(
            xaxis_title="Confidence",
            xaxis_range=[0, 1],
            height=300,
        )
        st.plotly_chart(fig, use_container_width=True)

    elif not review_text:
        st.info("Enter a review above to analyze it.")
        st.markdown("**Example reviews to try:**")
        examples = [
            "The battery died after 2 days. Customer service refused to help. Total waste of money.",
            "Arrived quickly and packaging was perfect. Easy to setup and works exactly as described!",
            "Quality is decent for the price but delivery took 3 weeks.",
        ]
        for ex in examples:
            st.markdown(f"> {ex}")
