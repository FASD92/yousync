import { NextRequest, NextResponse } from 'next/server';

export async function GET(req: NextRequest) {
  const { searchParams } = new URL(req.url);
  const url = searchParams.get('url');
  if (!url) {
    return NextResponse.json({ error: 'Missing url' }, { status: 400 });
  }
  try {
    const s3Res = await fetch(url);
    if (!s3Res.ok) {
      return NextResponse.json({ error: 'Failed to fetch S3 audio' }, { status: s3Res.status });
    }
    const contentType = s3Res.headers.get('content-type') || 'audio/mpeg';
    const arrayBuffer = await s3Res.arrayBuffer();
    return new NextResponse(Buffer.from(arrayBuffer), {
      status: 200,
      headers: {
        'Content-Type': contentType,
        'Cache-Control': 'public, max-age=60',
      },
    });
  } catch (e) {
    return NextResponse.json({ error: 'Proxy error' }, { status: 500 });
  }
} 