import { NextResponse } from 'next/server';

export async function GET(request, { params }) {
  const { state } = params;

  try {
    const res = await fetch(
      `https://kvkdev.pythonanywhere.com/state/${encodeURIComponent(state)}`
    );

    if (!res.ok) {
      return NextResponse.json(
        { error: `Upstream API error: HTTP ${res.status}` },
        { status: res.status }
      );
    }

    // The API returns bare `NaN` values which are invalid JSON.
    // We read as text first, replace NaN with null, then parse.
    const raw = await res.text();
    const sanitized = raw.replace(/:\s*NaN\b/g, ': null');
    const data = JSON.parse(sanitized);

    return NextResponse.json(data);
  } catch (err) {
    return NextResponse.json(
      { error: `Failed to fetch from upstream: ${err.message}` },
      { status: 502 }
    );
  }
}

