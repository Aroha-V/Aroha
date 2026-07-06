import { connectDB } from '@/lib/mongodb';
import { getTokenPayload } from '@/lib/auth';
import Conversation from '@/models/Conversation';

/** GET /api/conversations — list all conversations for the logged-in user */
export async function GET(request) {
  const payload = getTokenPayload(request);
  if (!payload) return Response.json({ error: 'Unauthorized' }, { status: 401 });

  await connectDB();

  const conversations = await Conversation.find({ userId: payload.id })
    .sort({ updatedAt: -1 })
    .select('_id title updatedAt')
    .lean();

  return Response.json({ conversations });
}

/** POST /api/conversations — create a new conversation */
export async function POST(request) {
  const payload = getTokenPayload(request);
  if (!payload) return Response.json({ error: 'Unauthorized' }, { status: 401 });

  const body = await request.json().catch(() => ({}));
  const title = (body.title || 'New Chat').slice(0, 80); // truncate long first messages

  await connectDB();

  const conv = await Conversation.create({ userId: payload.id, title });
  return Response.json({ conversation: { id: conv._id, title: conv.title } }, { status: 201 });
}
