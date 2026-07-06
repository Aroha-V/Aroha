import { getTokenPayload } from '@/lib/auth';
import { connectDB } from '@/lib/mongodb';
import Conversation from '@/models/Conversation';
import Message from '@/models/Message';

/**
 * POST /api/chat
 * Body: { message: string, conversationId?: string }
 *
 * - Always proxies to Flask at localhost:2000/Chatbot
 * - If user is authenticated AND conversationId provided → saves messages to MongoDB
 */
export async function POST(request) {
  try {
    const body = await request.json();
    const message = (body.message || '').trim();
    const conversationId = body.conversationId || null;

    if (!message) {
      return Response.json({ error: 'Empty message' }, { status: 400 });
    }

    // ── Proxy to Flask ───────────────────────────────────────────
    const fd = new FormData();
    fd.append('message', message);

    const flaskRes = await fetch('http://localhost:2000/Chatbot', {
      method: 'POST',
      body: fd,
    });

    if (!flaskRes.ok) {
      throw new Error(`Flask error: ${flaskRes.status} ${flaskRes.statusText}`);
    }

    const data = await flaskRes.json();
    const botText = typeof data === 'string' ? data : data.error || JSON.stringify(data);

    // ── Save to MongoDB if authenticated ─────────────────────────
    const payload = getTokenPayload(request);
    if (payload && conversationId) {
      try {
        await connectDB();

        // Verify the conversation belongs to this user
        const conv = await Conversation.findOne({
          _id: conversationId,
          userId: payload.id,
        });

        if (conv) {
          // Save user message
          await Message.create({ conversationId, role: 'user', text: message });
          // Save bot reply
          await Message.create({ conversationId, role: 'bot', text: botText });

          // Update conversation's updatedAt so it floats to top of sidebar
          await Conversation.findByIdAndUpdate(conversationId, { updatedAt: new Date() });
        }
      } catch (dbErr) {
        // DB errors should not break the chat response
        console.error('[chat/save]', dbErr.message);
      }
    }

    return Response.json(botText);
  } catch (err) {
    console.error('[/api/chat]', err.message);
    return Response.json(
      { error: err.message || 'Internal server error' },
      { status: 500 }
    );
  }
}
