const { createServer } = require('http');  // https → http 변경
const { parse }       = require('url');
const next            = require('next');

const dev = process.env.NODE_ENV !== 'production';
const hostname = '0.0.0.0';
const port     = process.env.PORT ? parseInt(process.env.PORT) : 3000;

const app = next({ dev, hostname, port });
const handle = app.getRequestHandler();

app.prepare().then(() => {
  createServer((req, res) => {
    const parsedUrl = parse(req.url, true);
    handle(req, res, parsedUrl);
  }).listen(port, hostname, err => {
    if (err) throw err;
    console.log(`> HTTP server running on http://localhost:${port}`);
  });
});