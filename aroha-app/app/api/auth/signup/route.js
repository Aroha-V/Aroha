import { connectDB } from '@/lib/mongodb';
import { signToken, setAuthCookie } from '@/lib/auth';
import User from '@/models/User';

export async function POST(request) {
  try {
    const { name, email, password } = await request.json();

    if (!name || !email || !password) {
      return Response.json({ error: 'All fields are required.' }, { status: 400 });
    }
    if (password.length < 6) {
      return Response.json({ error: 'Password must be at least 6 characters.' }, { status: 400 });
    }

    await connectDB();

    const existing = await User.findOne({ email });
    if (existing) {
      return Response.json({ error: 'An account with this email already exists.' }, { status: 409 });
    }

    const user = await User.create({ name, email, password });
    const token = signToken({ id: user._id, name: user.name, email: user.email });

    const response = Response.json({
      user: { id: user._id, name: user.name, email: user.email },
    });
    setAuthCookie(response, token);
    return response;
  } catch (err) {
    console.error('[signup]', err);
    return Response.json({ error: 'Server error. Please try again.' }, { status: 500 });
  }
}
