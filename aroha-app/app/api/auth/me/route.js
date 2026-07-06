import { getTokenPayload } from '@/lib/auth';

export async function GET(request) {
  const payload = getTokenPayload(request);
  if (!payload) {
    return Response.json({ user: null }, { status: 401 });
  }
  return Response.json({
    user: { id: payload.id, name: payload.name, email: payload.email },
  });
}
