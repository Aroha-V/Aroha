import { connectDB } from '@/lib/mongodb';
import { signToken, setAuthCookie } from '@/lib/auth';
import User from '@/models/User';

export async function POST(request) {
  try {
    const { email, password } = await request.json();

    if (!email || !password) {
      return Response.json({ error: 'Email and password are required.' }, { status: 400 });
    }

    await connectDB();

    const user = await User.findOne({ email });
    if (!user) {
      return Response.json({ error: 'Invalid email or password.' }, { status: 401 });
    }

    const match = await user.comparePassword(password);
    if (!match) {
      return Response.json({ error: 'Invalid email or password.' }, { status: 401 });
    }

    const token = signToken({ id: user._id, name: user.name, email: user.email });

    const response = Response.json({
      user: { id: user._id, name: user.name, email: user.email },
    });
    setAuthCookie(response, token);
    return response;
  } catch (err) {
    console.error('[login]', err);
    return Response.json({ error: 'Server error. Please try again.' }, { status: 500 });
  }
}
