import { clearAuthCookie } from '@/lib/auth';

export async function POST() {
  const response = Response.json({ ok: true });
  clearAuthCookie(response);
  return response;
}
