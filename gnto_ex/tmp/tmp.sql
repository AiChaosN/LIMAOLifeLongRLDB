-- 1. 开启 BAO 扩展的总开关
SET enable_bao TO on;

-- 2. 告诉 PG "我要你问问 AI 模型该选哪个计划"
SET enable_bao_selection TO on;

-- 3. (可选) 如果你不需要收集训练数据，可以关掉这个
SET enable_bao_rewards TO on;

-- 4. 关键：告诉 PG 生成多少个候选计划发给 Server (通常是 49)
SET bao_num_arms TO 49;

-- 5. 运行你的 SQL
SELECT count(*) FROM title t, movie_info mi WHERE t.id = mi.movie_id AND t.production_year > 2010;