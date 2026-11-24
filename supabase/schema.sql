-- ============================================================================
-- SUPABASE SCHEMA: Portfolio_Knowledge_Base (7-Tab AI Chatbot RAG Schema)
-- Target: Configure SUPABASE_URL in environment variables (.env)
-- Categories: Personal_Details, Work_Experience, Technical_Skills,
--             Projects, Articles, Social_Links, FAQs
-- Columns: id, category, title, content, url, active, sort_order, updated_at
--
-- Run once: Supabase Dashboard -> SQL Editor -> New query -> paste -> Run.
-- Safe to re-run. The categories must match app/models/knowledge.py.
-- ============================================================================

-- 1. Master table storing all 7 knowledge base categories
CREATE TABLE IF NOT EXISTS public.portfolio_knowledge_base (
  id TEXT PRIMARY KEY,
  category TEXT NOT NULL CHECK (
    category IN (
      'Personal_Details',
      'Work_Experience',
      'Technical_Skills',
      'Projects',
      'Articles',
      'Social_Links',
      'FAQs'
    )
  ),
  title TEXT NOT NULL DEFAULT '',
  content TEXT NOT NULL,
  url TEXT NOT NULL DEFAULT '',
  active BOOLEAN NOT NULL DEFAULT TRUE,
  sort_order INTEGER NOT NULL DEFAULT 0,
  updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Index for fast category + ordering queries (e.g. latest Projects on top)
CREATE INDEX IF NOT EXISTS idx_portfolio_kb_category_order
  ON public.portfolio_knowledge_base (category, sort_order ASC);

CREATE INDEX IF NOT EXISTS idx_portfolio_kb_active
  ON public.portfolio_knowledge_base (active);

-- Keep updated_at current on every edit (the admin grid and its "unsynced changes"
-- badge rely on it; the column default alone only covers inserts).
CREATE OR REPLACE FUNCTION public.portfolio_kb_touch_updated_at()
RETURNS TRIGGER
LANGUAGE plpgsql
SET search_path = ''
AS $$
BEGIN
  NEW.updated_at := NOW();
  RETURN NEW;
END;
$$;

DROP TRIGGER IF EXISTS portfolio_kb_touch_updated_at ON public.portfolio_knowledge_base;
CREATE TRIGGER portfolio_kb_touch_updated_at
  BEFORE UPDATE ON public.portfolio_knowledge_base
  FOR EACH ROW EXECUTE FUNCTION public.portfolio_kb_touch_updated_at();

-- 2. Enable Row Level Security (RLS)
ALTER TABLE public.portfolio_knowledge_base ENABLE ROW LEVEL SECURITY;

-- Allow public read access (publishable key) for external AI chatbot services
DROP POLICY IF EXISTS "Allow public read access on portfolio_knowledge_base" ON public.portfolio_knowledge_base;
CREATE POLICY "Allow public read access on portfolio_knowledge_base"
  ON public.portfolio_knowledge_base
  FOR SELECT
  USING (true);

-- Writes are server-only: the admin API uses the service_role / secret key, which
-- bypasses RLS. Public keys get read access and nothing else.
REVOKE INSERT, UPDATE, DELETE, TRUNCATE ON public.portfolio_knowledge_base FROM anon, authenticated;
GRANT SELECT, INSERT, UPDATE, DELETE ON public.portfolio_knowledge_base TO service_role;

-- 3. Dedicated Views matching each of the 7 Knowledge Base Tabs
--    Each view exposes exact columns: (id, title, content, url, active)
--
--    security_invoker = true: without it a Postgres view runs with its OWNER's rights,
--    bypassing RLS — and these simple views are auto-updatable, so the public key could
--    UPDATE/DELETE rows through /rest/v1/Projects. With it, the caller's RLS applies
--    (read-only for public keys). Requires Postgres 15+ (every current Supabase project).
CREATE OR REPLACE VIEW public."Personal_Details" WITH (security_invoker = true) AS
  SELECT id, title, content, url, CASE WHEN active THEN 'TRUE' ELSE 'FALSE' END AS active
  FROM public.portfolio_knowledge_base
  WHERE category = 'Personal_Details'
  ORDER BY sort_order ASC;

CREATE OR REPLACE VIEW public."Work_Experience" WITH (security_invoker = true) AS
  SELECT id, title, content, url, CASE WHEN active THEN 'TRUE' ELSE 'FALSE' END AS active
  FROM public.portfolio_knowledge_base
  WHERE category = 'Work_Experience'
  ORDER BY sort_order ASC;

CREATE OR REPLACE VIEW public."Technical_Skills" WITH (security_invoker = true) AS
  SELECT id, title, content, url, CASE WHEN active THEN 'TRUE' ELSE 'FALSE' END AS active
  FROM public.portfolio_knowledge_base
  WHERE category = 'Technical_Skills'
  ORDER BY sort_order ASC;

CREATE OR REPLACE VIEW public."Projects" WITH (security_invoker = true) AS
  SELECT id, title, content, url, CASE WHEN active THEN 'TRUE' ELSE 'FALSE' END AS active
  FROM public.portfolio_knowledge_base
  WHERE category = 'Projects'
  ORDER BY sort_order ASC;

CREATE OR REPLACE VIEW public."Articles" WITH (security_invoker = true) AS
  SELECT id, title, content, url, CASE WHEN active THEN 'TRUE' ELSE 'FALSE' END AS active
  FROM public.portfolio_knowledge_base
  WHERE category = 'Articles'
  ORDER BY sort_order ASC;

CREATE OR REPLACE VIEW public."Social_Links" WITH (security_invoker = true) AS
  SELECT id, title, content, url, CASE WHEN active THEN 'TRUE' ELSE 'FALSE' END AS active
  FROM public.portfolio_knowledge_base
  WHERE category = 'Social_Links'
  ORDER BY sort_order ASC;

CREATE OR REPLACE VIEW public."FAQs" WITH (security_invoker = true) AS
  SELECT id, title, content, url, CASE WHEN active THEN 'TRUE' ELSE 'FALSE' END AS active
  FROM public.portfolio_knowledge_base
  WHERE category = 'FAQs'
  ORDER BY sort_order ASC;

-- The views are for reading only.
REVOKE INSERT, UPDATE, DELETE ON
  public."Personal_Details", public."Work_Experience", public."Technical_Skills", public."Projects",
  public."Articles", public."Social_Links", public."FAQs"
  FROM anon, authenticated;

-- Make the new objects visible to the REST API immediately.
NOTIFY pgrst, 'reload schema';
