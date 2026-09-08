# Frontend image.
#
# NEXT_PUBLIC_* values are compiled into the browser bundle, not read at
# runtime, so the API URL and token have to be present when `next build` runs.
# That is why they are build args rather than environment variables on the
# running container -- setting them at `docker run` time would do nothing.
#
# It also means the token ends up inside the built image and inside the
# JavaScript the browser downloads. It protects the API from other processes
# and other machines, not from whoever is looking at the page.

FROM node:22-slim AS builder

WORKDIR /app

# Install against the lockfile first, so dependency layers survive source edits.
COPY frontend/package.json frontend/package-lock.json* ./
RUN npm ci

COPY frontend/ ./

ARG NEXT_PUBLIC_API_BASE_URL=http://localhost:8000
ARG NEXT_PUBLIC_API_AUTH_TOKEN=""
ENV NEXT_PUBLIC_API_BASE_URL=$NEXT_PUBLIC_API_BASE_URL \
    NEXT_PUBLIC_API_AUTH_TOKEN=$NEXT_PUBLIC_API_AUTH_TOKEN

RUN npm run build


FROM node:22-slim AS runner

WORKDIR /app
ENV NODE_ENV=production

# output: "standalone" emits a server with only the modules it imports, so this
# stage carries that rather than the full node_modules tree. static/ is not
# included in it and has to be copied alongside. There is no public/ in this
# project, so it is not copied -- COPY fails on a missing source.
COPY --from=builder /app/.next/standalone ./
COPY --from=builder /app/.next/static ./.next/static

# The node image ships an unprivileged user; nothing here needs root.
USER node

EXPOSE 3000
ENV PORT=3000 HOSTNAME=0.0.0.0

CMD ["node", "server.js"]
