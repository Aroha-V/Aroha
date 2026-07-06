/**
 * POST /api/auth/google
 *
 * Flow:
 * 1. Client does Google sign-in via Firebase → gets an ID token
 * 2. Client sends the ID token here
 * 3. We verify it with Firebase Admin
 * 4. We find-or-create the user in MongoDB
 * 5. We issue our own JWT (httpOnly cookie) — same as email/password auth
 */

import { connectDB } from '@/lib/mongodb';
import { signToken, setAuthCookie } from '@/lib/auth';
import { verifyFirebaseToken } from '@/lib/firebaseAdmin';
import User from '@/models/User';

export async function POST(request) {
  try {
    const { idToken } = await request.json();

    if (!idToken) {
      return Response.json({ error: 'Missing ID token.' }, { status: 400 });
    }

    // ── Verify the Firebase token ────────────────────────────────
    const decoded = await verifyFirebaseToken(idToken);
    if (!decoded) {
      return Response.json({ error: 'Invalid or expired Google token.' }, { status: 401 });
    }

    const { name, email, uid } = decoded;
    if (!email) {
      return Response.json({ error: 'Google account has no email.' }, { status: 400 });
    }

    // ── Find or create the user in MongoDB ───────────────────────
    await connectDB();

    let user = await User.findOne({ email });

    if (!user) {
      // New Google user — create account (no password needed)
      user = await User.create({
        name:     name || email.split('@')[0],
        email,
        password: uid, // Store Firebase UID as password placeholder (won't be used for login)
        provider: 'google',
      });
    }

    // ── Issue our JWT cookie (same as email/password login) ──────
    const token = signToken({ id: user._id, name: user.name, email: user.email });

    const response = Response.json({
      user: { id: user._id, name: user.name, email: user.email },
    });
    setAuthCookie(response, token);
    return response;
  } catch (err) {
    console.error('[auth/google]', err);
    return Response.json({ error: 'Server error. Please try again.' }, { status: 500 });
  }
}
