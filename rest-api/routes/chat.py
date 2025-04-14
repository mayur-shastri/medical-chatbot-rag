from flask import Blueprint, request, jsonify, session
from models.models import db, User, ChatHistory
from services.rag_chain import get_rag_response

chat_bp = Blueprint("chat", __name__, url_prefix="/chat")

@chat_bp.route("/ask", methods=["POST"])
def chat():
    user_id = session.get("user_id")
    if not user_id:
        return jsonify({"error": "Unauthorized"}), 401

    user = User.query.get(user_id)
    if not user:
        return jsonify({"error": "User not found"}), 404

    prompt = request.json.get("prompt")
    if not prompt:
        return jsonify({"error": "No prompt provided"}), 400

    # Get last N chat messages for context (can tweak how many)
    past_messages = ChatHistory.query.filter_by(user_id=user.id).order_by(ChatHistory.timestamp.desc()).limit(5).all()
    past_messages.reverse()  # So older messages come first

    chat_context = "\n".join(
        f"User: {msg.user_message}\nBot: {msg.bot_response}" for msg in past_messages
    )

    # Construct custom user context
    user_info = f"""
    Name: {user.name or 'Unknown'}
    Age: {user.age or 'Unknown'}
    Gender: {user.gender or 'Unknown'}
    Medical History: {user.medical_history or 'Not provided'}
    """

    # Combine all context and run RAG
    try:
        full_context = f"{user_info}\n\nPrevious Chat:\n{chat_context}"
        bot_response = get_rag_response(prompt, user_context=full_context)

        # Save chat to history
        new_chat = ChatHistory(
            user_id=user.id,
            user_message=prompt,
            bot_response=bot_response
        )
        db.session.add(new_chat)
        db.session.commit()

        return jsonify({
            "response": bot_response,
            "history_id": new_chat.id
        }), 200

    except Exception as e:
        return jsonify({"error": f"Chat processing failed: {str(e)}"}), 500
