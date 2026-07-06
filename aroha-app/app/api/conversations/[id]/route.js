import { connectDB } from '@/lib/mongodb';
import { getTokenPayload } from '@/lib/auth';
import Conversation from '@/models/Conversation';
import Message from '@/models/Message';

/** GET /api/conversations/[id] — load all messages for a conversation */
export async function GET(request, { params }) {
  const payload = getTokenPayload(request);
  if (!payload) return Response.json({ error: 'Unauthorized' }, { status: 401 });

  await connectDB();

  // Verify ownership
  const conv = await Conversation.findOne({ _id: params.id, userId: payload.id });
  if (!conv) return Response.json({ error: 'Not found' }, { status: 404 });

  const messages = await Message.find({ conversationId: params.id })
    .sort({ createdAt: 1 })
    .select('_id role text isError createdAt')
    .lean();

  return Response.json({ messages });
}

/** DELETE /api/conversations/[id] — delete conversation and all its messages */
export async function DELETE(request, { params }) {
  const payload = getTokenPayload(request);
  if (!payload) return Response.json({ error: 'Unauthorized' }, { status: 401 });

  await connectDB();

  const conv = await Conversation.findOne({ _id: params.id, userId: payload.id });
  if (!conv) return Response.json({ error: 'Not found' }, { status: 404 });

  await Message.deleteMany({ conversationId: params.id });
  await conv.deleteOne();

  return Response.json({ ok: true });
}
