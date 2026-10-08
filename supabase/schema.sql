create extension if not exists pgcrypto;

create table if not exists public.profiles (
    id uuid primary key references auth.users(id) on delete cascade,
    email text not null unique,
    display_name text not null default '',
    role text not null default 'user' check (role in ('user', 'admin')),
    created_at timestamptz not null default now()
);

create or replace function public.create_profile_for_new_user()
returns trigger
language plpgsql
security definer
set search_path = ''
as $$
begin
    insert into public.profiles (id, email, display_name)
    values (new.id, new.email, coalesce(new.raw_user_meta_data ->> 'display_name', ''))
    on conflict (id) do nothing;
    return new;
end;
$$;

drop trigger if exists on_auth_user_created_profile on auth.users;
create trigger on_auth_user_created_profile
    after insert on auth.users
    for each row execute procedure public.create_profile_for_new_user();

alter table public.profiles enable row level security;
revoke all on public.profiles from anon, authenticated;
grant select on public.profiles to authenticated;
grant update (display_name) on public.profiles to authenticated;
grant all on public.profiles to service_role;

drop policy if exists "Users can read their own profile" on public.profiles;
create policy "Users can read their own profile"
    on public.profiles for select to authenticated
    using ((select auth.uid()) = id);

drop policy if exists "Users can update their own display name" on public.profiles;
create policy "Users can update their own display name"
    on public.profiles for update to authenticated
    using ((select auth.uid()) = id)
    with check ((select auth.uid()) = id);

create table if not exists public.valuations (
    id uuid primary key default gen_random_uuid(),
    user_id uuid not null references public.profiles(id) on delete cascade,
    created_at timestamptz not null default now(),
    currency text not null,
    country text,
    city text,
    predicted_price numeric not null,
    price_formatted text not null,
    property_data jsonb not null,
    result_data jsonb not null
);

create index if not exists valuations_user_created_at_idx
    on public.valuations (user_id, created_at desc);
create index if not exists valuations_created_at_idx
    on public.valuations (created_at desc);

alter table public.valuations enable row level security;
revoke all on public.valuations from anon, authenticated;
grant select, insert, delete on public.valuations to authenticated;
grant all on public.valuations to service_role;

drop policy if exists "Users can read their own valuations" on public.valuations;
create policy "Users can read their own valuations"
    on public.valuations for select to authenticated
    using ((select auth.uid()) = user_id);

drop policy if exists "Users can save their own valuations" on public.valuations;
create policy "Users can save their own valuations"
    on public.valuations for insert to authenticated
    with check ((select auth.uid()) = user_id);

drop policy if exists "Users can delete their own valuations" on public.valuations;
create policy "Users can delete their own valuations"
    on public.valuations for delete to authenticated
    using ((select auth.uid()) = user_id);

create table if not exists public.market_settings (
    is_singleton boolean primary key default true check (is_singleton),
    enabled_countries text[] not null,
    enabled_currencies text[] not null,
    updated_at timestamptz not null default now(),
    updated_by uuid references public.profiles(id)
);

alter table public.market_settings enable row level security;
revoke all on public.market_settings from anon, authenticated;
grant select on public.market_settings to anon, authenticated;
grant all on public.market_settings to service_role;

drop policy if exists "Market allowlist is public" on public.market_settings;
create policy "Market allowlist is public"
    on public.market_settings for select to anon, authenticated
    using (true);