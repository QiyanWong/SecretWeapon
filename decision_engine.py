import time
import math
import os
import random

from map_parser import MapParser
from a_star_pathfinder import AStarPathfinder, NavAction, NavStep

class FSMState:
    IDLE = "IDLE"
    PATROLLING = "PATROLLING"
    COMBAT_MOVING = "COMBAT_MOVING"
    COMBAT = "COMBAT"
    RESTING = "RESTING"  # 连续 30s 无怪就近发呆待机

class DecisionEngine:
    def __init__(self, game_controller, route_manager):
        self.gc = game_controller
        self.route_manager = route_manager
        
        self.state = FSMState.IDLE
        self.last_state = FSMState.IDLE
        
        # 目标与状态追踪
        self.combat_target_pos = None    # 战斗锁定的怪物屏幕坐标
        self.combat_start_time = 0       # 战斗开始时间 (防超时卡死)
        
        # 传统模式爬绳状态追踪
        self.is_climbing_rope = False
        self.climb_last_y = -1
        self.climb_last_y_time = 0
        self.climb_start_time = 0        # 尝试挂绳的时间
        self.climb_attempt = 0           # 爬绳尝试次数
        self.climb_caught = False        # 是否已成功抓绳 (纵向坐标开始改变)
        
        # 高级 XML 地图拓扑寻路系统
        self.map_parser = None
        self.pathfinder = None
        self.active_path = []            # 当前执行中的 A* 路径步骤序列 [NavStep, ...]
        self.current_step_idx = 0        # 当前步骤索引
        self.patrol_targets = []         # 自动生成的全图主要平台巡逻目标
        self.patrol_idx = 0              # 巡逻目标索引
        self.last_replan_time = 0        # 防频繁重复规划
        self.climb_finish_time = 0       # 登顶持续按 UP 缓冲计时器
        self.last_jump_climb_time = 0    # 悬挂梯跳跃抓梯冷却计时器
        
        # 仲裁参数
        self.ATTACK_RANGE_X = 140        # 攻击范围 (X 像素距离)
        self.ATTACK_RANGE_Y = 70         # 攻击范围 (Y 像素高度差，放宽容忍度)
        self.MONSTER_AGRO_DIST = 400     # 接敌范围 (屏幕像素)
        self.MINIMAP_NODE_DIST = 8       # 小地图节点到达判定距离 (小地图像素)

        self.last_attack_time = 0
        self.last_normal_attack_time = 0
        self.last_aoe_attack_time = 0
        self.target_normal_atk_interval = 0.3 # 动态浮动普攻目标 CD
        self.target_aoe_skill_interval = 0.45  # 动态浮动群攻目标 CD
        self.first_monster_seen_time = 0.0     # 拟人化视觉感知缓冲计时戳
        self.reaction_delay_target = 0.16       # 动态认知反应时长 (120ms~220ms)
        self.last_state_change_time = 0  # 状态切换时间戳
        self.no_monster_start_time = 0   # 连续无怪计时器 (若期间有怪则清零)
        self.no_combat_start_time = 0.0  # 连续无战斗脱战计时器 (30s 触发就近发呆待机)
        self.no_combat_max_duration = 30.0 # 连续 30s 无战斗进入待机
        self.micro_idle_end_time = 0.0   # 巡逻路点微发呆结束时间戳 (1.0~2.0s)
        self.waypoint_spatial_jitter_x = 0 # 空间路点与折返点动态随机漂移 (±12px)
        self.last_log_time = 0           # 限流日志打印时间
        
        # 平台打怪与危险区防掉落避险系统
        self.current_danger_margin = 150   # 平台左右危险区避险边距 (px)
        self.platform_patrol_dir = "RIGHT" # 平台来回巡逻方向 ("RIGHT" / "LEFT")
        self.is_escaping_platform_danger = False # 危险区紧急回撤标志
        self.last_platform_danger_log_time = 0.0 # 危险区日志限流
        self.current_platform_bounds = None # (curr_fh, x_min, x_max, safe_left, safe_right, x_mid)

    def load_map_xml(self, xml_path):
        """加载解包地图 XML 并初始化 A* 拓扑寻路器"""
        if not xml_path or not os.path.exists(xml_path):
            print(f"【警告】地图 XML 文件不存在: {xml_path}")
            self.map_parser = None
            self.pathfinder = None
            return False
            
        try:
            self.map_parser = MapParser(xml_path)
            self.pathfinder = AStarPathfinder(self.map_parser)
            self.active_path = []
            self.current_step_idx = 0
            
            # 自动提取主要平台作为巡逻目标 (选取宽度 > 200px 的水平平台)
            major_fhs = [fh for fh in self.map_parser.horizontal_fhs if (fh.x2 - fh.x1) >= 150]
            if not major_fhs:
                major_fhs = self.map_parser.horizontal_fhs
                
            # 按 Y 高度排序，形成从下到上再到下的环路巡逻
            self.patrol_targets = []
            for fh in major_fhs:
                mid_x = int((fh.x1 + fh.x2) / 2)
                mid_y = int(fh.get_y_at_x(mid_x))
                self.patrol_targets.append((mid_x, mid_y, fh.id))
                
            self.patrol_idx = 0
            print(f"🗺️ 【高级寻路初始化成功】已提取 {len(self.patrol_targets)} 个主要平台巡逻航点")
            return True
        except Exception as e:
            print(f"【错误】解析地图 XML 失败: {e}")
            self.map_parser = None
            self.pathfinder = None
            return False

    def get_platform_info(self, x_world, y_world, danger_margin=150):
        """
        根据当前世界坐标定位玩家脚下所在的连续完整平台，并计算左右 150px 危险区边界
        返回: (curr_fh, x_min, x_max, safe_left, safe_right, x_mid) 或 None
        """
        if not self.map_parser:
            return None
            
        curr_fh = self.map_parser.snap_to_foothold(x_world, y_world, margin=60)
        if not curr_fh:
            return None
            
        visited = {curr_fh.id}
        queue = [curr_fh]
        connected = [curr_fh]
        
        while queue:
            f = queue.pop(0)
            for o in self.map_parser.horizontal_fhs:
                if o.id in visited:
                    continue
                # 检查连续相接 (端点横向贴合 <= 15px, 纵向落差 <= 20px)
                if (abs(o.x2 - f.x1) <= 15 and abs(o.y2 - f.y1) <= 20) or \
                   (abs(o.x1 - f.x2) <= 15 and abs(o.y1 - f.y2) <= 20):
                    visited.add(o.id)
                    queue.append(o)
                    connected.append(o)
                    
        x_min = min(min(f.x1, f.x2) for f in connected)
        x_max = max(max(f.x1, f.x2) for f in connected)
        total_w = x_max - x_min
        
        # 针对极窄平台做动态收缩保护 (避免窄平台上安全区消失)
        actual_margin = danger_margin
        if total_w < (danger_margin * 2 + 40):
            actual_margin = max(20, int(total_w * 0.25))
            
        safe_left = x_min + actual_margin
        safe_right = x_max - actual_margin
        x_mid = (x_min + x_max) / 2.0
        
        return curr_fh, x_min, x_max, safe_left, safe_right, x_mid

    def update(self, game_screen_player_pos, game_screen_monsters, player_state, 
               minimap_player_pos, config):
        """
        每帧调用的核心决策逻辑
        """
        if not minimap_player_pos:
            self.gc.clear_movement()
            return

        # 屏幕玩家位置记忆与受击闪烁兜底 (若受击扣血闪烁无敌帧导致单帧未检出，维持当前记忆坐标)
        if game_screen_player_pos:
            self.last_known_screen_player_pos = game_screen_player_pos
        else:
            game_screen_player_pos = self.last_known_screen_player_pos

        # 拟人化鼠标微扰周期检查 (打破 30 分钟 0 鼠标特征)
        if hasattr(self.gc, 'mouse_drifter') and self.gc.mouse_drifter:
            self.gc.mouse_drifter.tick()

        now = time.time()
        self.MONSTER_AGRO_DIST = config.get("monster_agro_dist", 500)
        self.ATTACK_RANGE_Y = config.get("attack_range_y", 120)
        self.state_switch_interval = config.get("state_switch_interval", 0.5)

        # 🌟 平台打怪模式：最高优先级危险区防掉落避险拦截
        enable_platform_patrol = config.get("enable_platform_patrol", False)
        crop_w = config.get("crop_w", None)
        crop_h = config.get("crop_h", None)
        danger_margin = config.get("danger_margin", 150)
        self.current_danger_margin = danger_margin
        
        if enable_platform_patrol and self.map_parser:
            mx, my = minimap_player_pos
            curr_xw, curr_yw = self.map_parser.minimap_to_world(mx, my, crop_w=crop_w, crop_h=crop_h)
            p_info = self.get_platform_info(curr_xw, curr_yw, danger_margin=danger_margin)
            if p_info:
                curr_fh, x_min, x_max, safe_left, safe_right, x_mid = p_info
                self.current_platform_bounds = p_info
                
                # (1) 落入左侧危险区 (X < safe_left): 强制向右回撤中点，并同步巡逻方向为向右
                if curr_xw < safe_left:
                    self.is_escaping_platform_danger = True
                    self.platform_patrol_dir = "RIGHT"
                    self.waypoint_spatial_jitter_x = random.randint(0, 8)
                    self.gc.release_key("LEFT")
                    self.gc.press_key("RIGHT")
                    
                    # 若当前处于战斗状态，立即强制退出战斗
                    if self.state in [FSMState.COMBAT, FSMState.COMBAT_MOVING]:
                        self._set_state(FSMState.PATROLLING)
                        self.combat_target_pos = None
                        self.combat_start_time = 0
                        
                    if now - self.last_platform_danger_log_time >= 1.2:
                        self.last_platform_danger_log_time = now
                        print(f"🚨 [平台避险] 玩家落入左侧 {danger_margin}px 危险区 (X={curr_xw:.0f} < 安全界限 {safe_left:.0f})！强制退出战斗，往平台中点 ({x_mid:.0f}) 紧急回退！")
                    return
                    
                # (2) 落入右侧危险区 (X > safe_right): 强制向左回撤中点，并同步巡逻方向为向左
                elif curr_xw > safe_right:
                    self.is_escaping_platform_danger = True
                    self.platform_patrol_dir = "LEFT"
                    self.waypoint_spatial_jitter_x = random.randint(0, 8)
                    self.gc.release_key("RIGHT")
                    self.gc.press_key("LEFT")
                    
                    # 若当前处于战斗状态，立即强制退出战斗
                    if self.state in [FSMState.COMBAT, FSMState.COMBAT_MOVING]:
                        self._set_state(FSMState.PATROLLING)
                        self.combat_target_pos = None
                        self.combat_start_time = 0
                        
                    if now - self.last_platform_danger_log_time >= 1.2:
                        self.last_platform_danger_log_time = now
                        print(f"🚨 [平台避险] 玩家落入右侧 {danger_margin}px 危险区 (X={curr_xw:.0f} > 安全界限 {safe_right:.0f})！强制退出战斗，往平台中点 ({x_mid:.0f}) 紧急回退！")
                    return
                else:
                    self.is_escaping_platform_danger = False
            else:
                self.current_platform_bounds = None
                self.is_escaping_platform_danger = False
        else:
            self.current_platform_bounds = None
            self.is_escaping_platform_danger = False

        # 1. 查找屏幕内符合攻击距离与群攻门槛的怪物
        closest_same_level_monster = None
        closest_same_level_dist = 9999
        closest_all_monster = None
        min_all_dist = 9999
        same_level_monsters = []

        normal_atk_range = config.get("normal_atk_range", 140)
        aoe_skill_range = config.get("aoe_skill_range", 200)
        aoe_monster_count = config.get("aoe_monster_count", 3)
        aoe_dir_mode = config.get("aoe_dir_mode", "单向 (单侧面朝方向)")
        is_single_dir = ("单向" in aoe_dir_mode)
        self.ATTACK_RANGE_X = normal_atk_range

        px = game_screen_player_pos[0]
        py = game_screen_player_pos[1]
        p_bottom = game_screen_player_pos[5] if len(game_screen_player_pos) >= 6 else (py + 30)

        if game_screen_monsters:
            for m in game_screen_monsters:
                mx = m[0]
                my = m[1]
                m_bottom = m[5] if len(m) >= 6 else my
                
                dist = math.hypot(mx - px, my - py)
                if dist < min_all_dist:
                    min_all_dist = dist
                    closest_all_monster = (mx, my)

                # 双重高度对齐判定：中心点高度差 与 底部脚底高度差 取最优
                y_center_diff = abs(my - py)
                y_bottom_diff = abs(m_bottom - p_bottom) if (len(m) >= 6 and len(game_screen_player_pos) >= 6) else y_center_diff
                effective_y_diff = min(y_center_diff, y_bottom_diff)

                # 筛选同平台/同高度差在 attack_range_y 像素内的有效怪物
                if effective_y_diff <= self.ATTACK_RANGE_Y:
                    same_level_monsters.append(m)
                    if dist < closest_same_level_dist:
                        closest_same_level_dist = dist
                        closest_same_level_monster = (mx, my)

        # 统计在群攻范围与普攻范围内的同高度怪物数
        if is_single_dir:
            r_monsters = [m for m in same_level_monsters if 0 <= (m[0] - px) <= aoe_skill_range]
            l_monsters = [m for m in same_level_monsters if 0 <= (px - m[0]) <= aoe_skill_range]
            if len(r_monsters) >= aoe_monster_count:
                aoe_monsters = r_monsters
            elif len(l_monsters) >= aoe_monster_count:
                aoe_monsters = l_monsters
            else:
                aoe_monsters = []
        else:
            aoe_monsters = [m for m in same_level_monsters if abs(m[0] - px) <= aoe_skill_range]

        normal_monsters = [m for m in same_level_monsters if abs(m[0] - px) <= normal_atk_range]

        # 2. 状态跃迁逻辑 (FSM) - 单向无怪静默防抖：寻路遇怪进战斗；战斗无怪开始计时，达到设定间隔仍无怪切回寻路
        if self.state == FSMState.IDLE:
            self._set_state(FSMState.PATROLLING)

        # 攀爬锁：仅在真正攀爬绳子/梯子且非平台打怪模式时触发
        is_climbing = self.is_climbing_rope or (player_state == "climb" and not enable_platform_patrol and not self.current_platform_bounds)
        if is_climbing and self.state in [FSMState.COMBAT, FSMState.COMBAT_MOVING]:
            print("【攀爬锁触发】正在攀爬绳子/梯子，暂停打怪状态，专注登顶！")
            self._set_state(FSMState.PATROLLING)
            self.no_monster_start_time = 0

        # 判断同高度/同平台寻怪范围内是否有处于有效攻击区间的怪
        has_agro_monster = (closest_same_level_monster is not None and closest_same_level_dist <= self.MONSTER_AGRO_DIST)
        has_attack_targets = (len(aoe_monsters) >= aoe_monster_count or len(normal_monsters) > 0)

        # 核心机制：只要范围内检测到同平台有效怪物，无怪静默计时器与脱战发呆计时器立即清零！
        if has_agro_monster or has_attack_targets:
            self.no_monster_start_time = 0
            self.no_combat_start_time = 0
            self.micro_idle_end_time = 0  # 刷怪立即打断微发呆

        # 规则 0: 处于长时间无怪就近发呆待机状态 (RESTING)
        if self.state == FSMState.RESTING:
            if has_attack_targets:
                print(f"[RESTING -> COMBAT] 待机发呆中发现周围怪物进入攻击范围！立刻唤醒，切入战斗！")
                self.no_combat_start_time = 0
                self._set_state(FSMState.COMBAT)
                self.combat_start_time = now
                self.combat_target_pos = None
            elif has_agro_monster:
                print(f"[RESTING -> COMBAT_MOVING] 待机发呆中发现周围刷新怪物！立刻唤醒，切入追击走位！")
                self.no_combat_start_time = 0
                self._set_state(FSMState.COMBAT_MOVING)
                self.combat_target_pos = closest_same_level_monster
            else:
                # 保持原地待机发呆，等待怪物刷新
                self.gc.clear_movement()

        # 规则 A: 正在寻路/巡逻 (PATROLLING)
        elif self.state == FSMState.PATROLLING and not is_climbing:
            if has_attack_targets:
                # 1. 怪物已在有效攻击范围内：立刻切入 COMBAT 模式攻击！
                self.no_combat_start_time = 0
                self.first_monster_seen_time = 0
                self._set_state(FSMState.COMBAT)
                self.combat_start_time = now
                self.combat_target_pos = None
            elif has_agro_monster:
                # 2. 范围内发现同高度怪物：立刻切入 COMBAT_MOVING 走位追击！
                self.no_combat_start_time = 0
                self.first_monster_seen_time = 0
                self._set_state(FSMState.COMBAT_MOVING)
                self.combat_target_pos = closest_same_level_monster
            else:
                self.first_monster_seen_time = 0
                # 平台打怪模式下不触发 30s 停止巡逻发呆，保持左右两端持续巡逻
                if enable_platform_patrol:
                    self.no_combat_start_time = 0
                else:
                    # 30 秒连续无战斗脱战检测：如果持续 30 秒没有遇到怪，直接就近在当前节点发呆待机
                    if self.no_combat_start_time == 0:
                        self.no_combat_start_time = now
                    elif (now - self.no_combat_start_time) >= self.no_combat_max_duration:
                        print(f"[PATROL -> RESTING] 连续 {now - self.no_combat_start_time:.1f}s 未进入战斗，停止左右重复巡逻，就近在当前节点发呆待机...")
                        self.gc.clear_movement()
                        self._set_state(FSMState.RESTING)
                        self.no_combat_start_time = 0

        # 规则 B: 正在战斗输出 (COMBAT)
        elif self.state == FSMState.COMBAT and not is_climbing:
            self.no_combat_start_time = 0
            if has_attack_targets:
                # 攻击范围内有怪，持续攻击 (计时已在上方清零)
                self.no_monster_start_time = 0
            elif has_agro_monster:
                # 攻击范围内怪被打退/走动，但同平台仍有怪在寻怪范围内，立刻接近
                self._set_state(FSMState.COMBAT_MOVING)
                self.combat_target_pos = closest_same_level_monster
                self.no_monster_start_time = 0
            else:
                # 范围内无怪：直接开始计时，如果期间有怪直接清零，直到达到设定间隔仍然无怪才切回寻路
                if self.no_monster_start_time == 0:
                    self.no_monster_start_time = now

                elapsed = now - self.no_monster_start_time
                if elapsed >= self.state_switch_interval:
                    print(f"[COMBAT] 连续无怪达到设定间隔 {elapsed:.2f}s >= {self.state_switch_interval:.2f}s，切回寻路模式")
                    self._set_state(FSMState.PATROLLING)
                    self.no_monster_start_time = 0
                else:
                    # 计时期间原地保持警戒，不走动
                    self.gc.clear_movement()

            # 超时兜底：打同一处怪超过 8 秒强制解除
            if self.combat_start_time and (now - self.combat_start_time > 8.0):
                self._set_state(FSMState.PATROLLING)
                self.no_monster_start_time = 0

        # 规则 C: 正在走位接近怪物 (COMBAT_MOVING)
        elif self.state == FSMState.COMBAT_MOVING and not is_climbing:
            self.no_combat_start_time = 0
            if has_attack_targets:
                # 进入了攻击范围，立刻转入输出
                self._set_state(FSMState.COMBAT)
                self.combat_start_time = now
                self.combat_target_pos = None
                self.no_monster_start_time = 0
            elif has_agro_monster:
                # 目标怪仍在寻怪范围，继续向其移动
                self.combat_target_pos = closest_same_level_monster
                self.no_monster_start_time = 0
            else:
                # 追击目标丢失且范围内无怪 -> 同样执行无怪静默计时
                if self.no_monster_start_time == 0:
                    self.no_monster_start_time = now

                elapsed = now - self.no_monster_start_time
                if elapsed >= self.state_switch_interval:
                    print(f"[COMBAT_MOVING] 追击目标丢失且连续无怪达到 {elapsed:.2f}s >= {self.state_switch_interval:.2f}s，切回寻路模式")
                    self._set_state(FSMState.PATROLLING)
                    self.no_monster_start_time = 0
                else:
                    self.gc.clear_movement()

        # 限流 Debug 打印
        if now - self.last_log_time > 1.0:
            same_cnt = len(same_level_monsters)
            tot_cnt = len(game_screen_monsters) if game_screen_monsters else 0
            same_dist_str = f"{closest_same_level_dist:.1f}" if closest_same_level_monster else "None"
            print(f"[DECISION DEBUG] State={self.state}, TotalMonsters={tot_cnt}, SameLevelMonsters={same_cnt}, ClosestSameLevelDist={same_dist_str}, AllClosestDist={min_all_dist:.1f}, AoEMonsters={len(aoe_monsters)}, NormalMonsters={len(normal_monsters)}")
            self.last_log_time = now

        # 3. 状态动作执行 (Action Execution)
        if self.state == FSMState.PATROLLING:
            self._execute_patrol(minimap_player_pos, player_state, config)
        elif self.state == FSMState.RESTING:
            self.gc.clear_movement()
        elif self.state == FSMState.COMBAT_MOVING:
            self._execute_combat_moving(game_screen_player_pos, closest_same_level_monster or self.combat_target_pos)
        elif self.state == FSMState.COMBAT:
            self._execute_combat(game_screen_player_pos, same_level_monsters, config)

    def _set_state(self, new_state, force=False):
        if self.state != new_state or force:
            now = time.time()
            duration = now - self.last_state_change_time if self.last_state_change_time > 0 else 0
            if self.state != new_state:
                print(f"[FSM 状态切换] {self.state} ---> {new_state} (上一状态持续 {duration:.2f}s)")
            self.state = new_state
            self.last_state_change_time = now
            return True
        return False

    def _execute_patrol(self, m_pos, player_state, config):
        # 微发呆检查：如果当前处于路点微发呆中 (1.0~2.0s)，原地待机
        now = time.time()
        if now < self.micro_idle_end_time:
            self.gc.clear_movement()
            return

        # 模式分流 1: 若勾选了【平台打怪】，直接在当前平台的左右安全边界间来回巡逻
        if config.get("enable_platform_patrol", False) and self.map_parser is not None:
            self._execute_platform_patrol(m_pos, player_state, config)
            return

        # 模式分流 2: 若启用了高级 XML 寻路且已成功载入地图拓扑，走高级 A* 驱动
        if config.get("enable_advanced_nav", False) and self.pathfinder is not None:
            self._execute_advanced_patrol(m_pos, player_state, config)
            return

        # 【传统模式】100% 保留现有路点与走廊巡逻逻辑
        mx, my = m_pos
        nodes = self.route_manager.nodes
        n_count = len(nodes)
        if n_count == 0:
            self.gc.clear_movement()
            return
        
        curr_idx = self.route_manager.current_target_index
        if curr_idx >= n_count:
            self.route_manager.current_target_index = 0
            curr_idx = 0

        # 当前在有向线段 S_i (P_i -> P_{i+1}) 中，目的地为下一个点 P_{i+1}
        next_idx = (curr_idx + 1) % n_count if n_count > 1 else 0
        target_node = nodes[next_idx]

        # 如果当前正在执行爬绳的长程动作，拦截普通的巡路判定
        if self.is_climbing_rope:
            self._handle_climb_action(m_pos, player_state, config)
            return

        # 1. 检查玩家坐标是否处于当前有向路段 (P_i -> P_{i+1}) 的矩形走廊内
        in_corridor = self.route_manager.is_in_directed_segment_rectangle(curr_idx, mx, my)

        # 2. 如果脱离了当前矩形，全图搜寻玩家落入的走廊矩形，或重吸附到最近节点
        if not in_corridor and n_count > 1:
            found_rect_idx = None
            for idx in range(n_count):
                if self.route_manager.is_in_directed_segment_rectangle(idx, mx, my):
                    found_rect_idx = idx
                    break
            
            if found_rect_idx is not None:
                self.route_manager.current_target_index = found_rect_idx
                curr_idx = found_rect_idx
                next_idx = (curr_idx + 1) % n_count
                target_node = nodes[next_idx]
            else:
                nearest_idx = self.route_manager.find_nearest_node_index(mx, my)
                self.route_manager.current_target_index = nearest_idx
                curr_idx = nearest_idx
                next_idx = (curr_idx + 1) % n_count
                target_node = nodes[next_idx]

        # 3. 走廊内前进与终点 P_{i+1} 抵达判断
        tx, ty = target_node.x + self.waypoint_spatial_jitter_x, target_node.y
        dist = math.hypot(tx - mx, ty - my)

        if dist < self.MINIMAP_NODE_DIST or (abs(tx - mx) <= 3 and abs(ty - my) <= 5):
            if target_node.action_type == "CLIMB":
                print(f"[PATROL] 抵达爬绳节点 #{target_node.node_id}，开始进入爬绳模式！")
                self.is_climbing_rope = True
                self.climb_caught = False
                self.climb_last_y = my
                self.climb_last_y_time = time.time()
                self.climb_start_time = time.time()
                self.climb_attempt = 0
                self._handle_climb_action(m_pos, player_state, config)
                return
            else:
                self.route_manager.current_target_index = next_idx
                curr_idx = next_idx
                next_idx = (curr_idx + 1) % n_count
                target_node = nodes[next_idx]
                # 重新生成下次路点空间随机漂移 (±12px)，消除固定点折返
                self.waypoint_spatial_jitter_x = random.randint(-12, 12)
                tx = target_node.x + self.waypoint_spatial_jitter_x
                
                # 拟人化微发呆：25% 概率在抵达路点后略作 1.0 ~ 2.0 秒停顿，避免机械式秒转
                if random.random() < 0.25:
                    self.micro_idle_end_time = now + random.uniform(1.0, 2.0)
                    self.gc.clear_movement()
                    return

        # 4. 走向目标终点
        if target_node:
            self._move_towards_minimap(mx, tx, target_node.action_type, config)

    def _execute_platform_patrol(self, m_pos, player_state, config):
        """
        平台打怪专用巡逻器：
        无视一切录制航点与多跳 A* 路径，仅在当前平台的左右危险区边界节点之间来回巡逻
        """
        now = time.time()
        if now < self.micro_idle_end_time:
            self.gc.clear_movement()
            return
            
        mx, my = m_pos
        crop_w = config.get("crop_w", None)
        crop_h = config.get("crop_h", None)
        danger_margin = config.get("danger_margin", 150)
        curr_xw, curr_yw = self.map_parser.minimap_to_world(mx, my, crop_w=crop_w, crop_h=crop_h)
        
        p_info = self.get_platform_info(curr_xw, curr_yw, danger_margin=danger_margin)
        if not p_info:
            self.gc.clear_movement()
            return
            
        curr_fh, x_min, x_max, safe_left, safe_right, x_mid = p_info
        
        # 确保安全区宽度合理
        safe_w = max(10, safe_right - safe_left)
        # 折返缓冲：避免紧贴危险区边缘触发误判拦截，留出 10~20px 的平滑折返余量
        turn_buffer = min(20, max(5, int(safe_w * 0.08)))
        jitter = abs(self.waypoint_spatial_jitter_x) % 8
        
        target_left = safe_left + turn_buffer + jitter
        target_right = safe_right - turn_buffer - jitter
        if target_left >= target_right:
            target_left = safe_left + 5
            target_right = safe_right - 5
        
        # 向右巡逻
        if self.platform_patrol_dir == "RIGHT":
            if curr_xw >= target_right:
                # 抵达右侧安全边界！折返向左
                self.platform_patrol_dir = "LEFT"
                self.waypoint_spatial_jitter_x = random.randint(0, 8)
                if random.random() < 0.20:
                    self.micro_idle_end_time = now + random.uniform(0.6, 1.2)
                    self.gc.clear_movement()
                    return
                self.gc.release_key("RIGHT")
                self.gc.press_key("LEFT")
            else:
                self.gc.release_key("LEFT")
                self.gc.press_key("RIGHT")
        # 向左巡逻
        else:
            if curr_xw <= target_left:
                # 抵达左侧安全边界！折返向右
                self.platform_patrol_dir = "RIGHT"
                self.waypoint_spatial_jitter_x = random.randint(0, 8)
                if random.random() < 0.20:
                    self.micro_idle_end_time = now + random.uniform(0.6, 1.2)
                    self.gc.clear_movement()
                    return
                self.gc.release_key("LEFT")
                self.gc.press_key("RIGHT")
            else:
                self.gc.release_key("RIGHT")
                self.gc.press_key("LEFT")

    def _execute_advanced_patrol(self, m_pos, player_state, config):
        """
        高级 XML 地图拓扑 A* 巡逻驱动器
        支持融入用户录制的巡逻路线（优先提取 WALK 关键路点进行循环巡逻）
        """
        if not self.map_parser or not self.pathfinder:
            self.gc.clear_movement()
            return

        now = time.time()
        if now < self.micro_idle_end_time:
            self.gc.clear_movement()
            return

        mx, my = m_pos
        crop_w = config.get("crop_w", None)
        crop_h = config.get("crop_h", None)
        curr_xw, curr_yw = self.map_parser.minimap_to_world(mx, my, crop_w=crop_w, crop_h=crop_h)
        jump_key = config.get("jump_key", "Alt")

        # 1. 确定巡逻目标点列表 (优先使用用户录制的 WALK 节点，无录制时使用全图主要平台)
        targets = []
        user_nodes = self.route_manager.nodes
        if user_nodes:
            # 提取所有包含 WALK 意图的节点 (或全部录制节点)
            walk_nodes = [n for n in user_nodes if n.action_type in ["WALK", "WALK_SEGMENT"]]
            if not walk_nodes:
                walk_nodes = user_nodes
            for n in walk_nodes:
                wx, wy = self.map_parser.minimap_to_world(n.x, n.y, crop_w=crop_w, crop_h=crop_h)
                targets.append((wx, wy, f"录制路点 #{n.node_id} ({n.x},{n.y})"))
        elif self.patrol_targets:
            targets = self.patrol_targets

        if not targets:
            self.gc.clear_movement()
            return

        if self.patrol_idx >= len(targets):
            self.patrol_idx = 0

        target_pt = targets[self.patrol_idx]
        target_xw, target_yw, target_label = target_pt

        # 1.5. 检查是否已物理抵达当前巡逻目标点 (水平偏差 <= 30 且垂直落差 <= 45)
        dist_to_target = math.hypot(curr_xw - target_xw, curr_yw - target_yw)
        if dist_to_target <= 35 or (abs(curr_xw - target_xw) <= 30 and abs(curr_yw - target_yw) <= 45):
            # 已抵达当前目标点！切至下一个巡逻目标 (循环推进)
            self.patrol_idx = (self.patrol_idx + 1) % len(targets)
            self.active_path = []
            self.current_step_idx = 0
            self.climb_finish_time = 0
            next_target = targets[self.patrol_idx]
            print(f"🎯 [ADVANCED NAV] 成功抵达目标 [{target_label}]！当前玩家: 世界({curr_xw:.0f}, {curr_yw:.0f}) 小地图({mx}, {my}) | 目标: 世界({target_xw:.0f}, {target_yw:.0f}) | 自动切换至下一路点 [{next_target[2]}]")
            
            # 拟人化微发呆：25% 概率在抵达路点后随机微发呆 1.0 ~ 2.0 秒
            if random.random() < 0.25:
                self.micro_idle_end_time = now + random.uniform(1.0, 2.0)

            self.gc.clear_movement()
            return

        # 2. 检查当前是否需要规划新路径 (无路径或已走完当前路径)
        if not self.active_path or self.current_step_idx >= len(self.active_path):
            self.active_path = self.pathfinder.find_path(curr_xw, curr_yw, target_xw, target_yw)
            self.current_step_idx = 0
            self.climb_finish_time = 0
            print(f"[ADVANCED A* NAV] 规划前往目标 [{target_label}] | 玩家当前: 世界({curr_xw:.0f}, {curr_yw:.0f}) 小地图({mx}, {my}) -> 终点: 世界({target_xw:.0f}, {target_yw:.0f}) | 步骤数: {len(self.active_path)}")

        if not self.active_path or self.current_step_idx >= len(self.active_path):
            # 如果寻路返回空，直接尝试下一个目标
            self.patrol_idx = (self.patrol_idx + 1) % len(targets)
            self.active_path = []
            self.current_step_idx = 0
            self.gc.clear_movement()
            return

        # 3. 执行当前步骤 (NavStep)
        step = self.active_path[self.current_step_idx]
        action = step.action
        tx_world = step.target_x
        ty_world = step.target_y

        if action == NavAction.WALK:
            self.climb_finish_time = 0
            # 水平行走
            if abs(curr_xw - tx_world) > 30: # 距离目标 > 30 像素
                if curr_xw < tx_world:
                    self.gc.release_key("LEFT")
                    self.gc.press_key("RIGHT")
                else:
                    self.gc.release_key("RIGHT")
                    self.gc.press_key("LEFT")
            else:
                self.gc.clear_movement()
                self.current_step_idx += 1
                # 如果这是本段路径最后一步，且确认垂直高度已到达，立即切下一路点
                if self.current_step_idx >= len(self.active_path):
                    if abs(curr_yw - target_yw) <= 45:
                        self.patrol_idx = (self.patrol_idx + 1) % len(targets)
                        self.active_path = []
                        self.current_step_idx = 0
                        next_target = targets[self.patrol_idx]
                        print(f"🎯 [ADVANCED NAV] 完成路段！当前玩家: 世界({curr_xw:.0f}, {curr_yw:.0f}) 小地图({mx}, {my}) | 目标: 世界({target_xw:.0f}, {target_yw:.0f}) | 切换至下一路点 [{next_target[2]}]")
                    else:
                        # 仍不在目标高度层，重置路径重新规划跨层动作
                        self.active_path = []
                        self.current_step_idx = 0

        elif action == NavAction.CLIMB_UP:
            # 攀爬向上：若 X 偏差大先对齐梯子 X
            if abs(curr_xw - tx_world) > 15 and (player_state != "climb"):
                self.climb_finish_time = 0
                if curr_xw < tx_world:
                    self.gc.release_key("LEFT")
                    self.gc.press_key("RIGHT")
                else:
                    self.gc.release_key("RIGHT")
                    self.gc.press_key("LEFT")
            else:
                self.gc.release_key("LEFT")
                self.gc.release_key("RIGHT")

                # 获取绳子/梯子底部世界 Y 坐标
                y_bottom = step.extra_info.get("y_bottom", None)
                if y_bottom is None and self.map_parser:
                    for lr in self.map_parser.ladder_ropes:
                        if abs(lr.x - tx_world) <= 10 and abs(lr.y_top - ty_world) <= 15:
                            y_bottom = lr.y_bottom
                            break
                if y_bottom is None:
                    y_bottom = ty_world + 100

                # 判断绳子起点(底部)是否高于地面 20px 以上 (世界坐标中，Y越小越靠上，所以 curr_yw - y_bottom > 20 代表高于脚底地面 >20px)
                is_hanging = (curr_yw - y_bottom > 20)

                # 若为悬挂绳索且尚未抓上绳子 (处于地面/空中且高度在绳底下方)
                if is_hanging and (player_state != "climb") and (curr_yw > y_bottom + 5):
                    now_t = time.time()
                    if now_t - self.last_jump_climb_time > 0.45:
                        self.last_jump_climb_time = now_t
                        self.gc.tap_key(jump_key, 0.08)
                    self.gc.press_key("UP")
                else:
                    # 绳子触地或已在绳子上：直接按住 UP 爬行
                    self.gc.press_key("UP")
                
                # 登顶判断：当身体已完全登顶 (curr_yw <= ty_world + 5)
                if curr_yw <= ty_world + 5:
                    # 持续按住 UP 键 0.5 秒容错，确保角色完全站上平台翻身
                    if self.climb_finish_time == 0:
                        self.climb_finish_time = time.time()
                    elif time.time() - self.climb_finish_time >= 0.5:
                        self.gc.release_key("UP")
                        self.climb_finish_time = 0
                        self.current_step_idx += 1
                else:
                    self.climb_finish_time = 0

        elif action == NavAction.CLIMB_DOWN:
            self.climb_finish_time = 0
            # 攀爬向下
            if abs(curr_xw - tx_world) > 15 and (player_state != "climb"):
                if curr_xw < tx_world:
                    self.gc.release_key("LEFT")
                    self.gc.press_key("RIGHT")
                else:
                    self.gc.release_key("RIGHT")
                    self.gc.press_key("LEFT")
            else:
                self.gc.release_key("LEFT")
                self.gc.release_key("RIGHT")
                self.gc.press_key("DOWN")
                if curr_yw >= ty_world - 10:
                    self.gc.release_key("DOWN")
                    self.current_step_idx += 1

        elif action == NavAction.DROP_DOWN:
            # 下跳
            if abs(curr_xw - tx_world) > 30:
                if curr_xw < tx_world:
                    self.gc.release_key("LEFT")
                    self.gc.press_key("RIGHT")
                else:
                    self.gc.release_key("RIGHT")
                    self.gc.press_key("LEFT")
            else:
                self.gc.clear_movement()
                self.gc.press_key("DOWN")
                self.gc.tap_key(jump_key, 0.1)
                self.gc.release_key("DOWN")
                self.current_step_idx += 1

        elif action == NavAction.JUMP:
            # 边缘跳跃
            if curr_xw < tx_world:
                self.gc.press_key("RIGHT")
                self.gc.tap_key(jump_key, 0.1)
            else:
                self.gc.press_key("LEFT")
                self.gc.tap_key(jump_key, 0.1)
            self.current_step_idx += 1

    def _move_towards_minimap(self, current_x, target_x, action_type, config):
        """控制左右移动，并在需要时触发跳跃/动作"""
        jump_key = config.get("jump_key", "Alt")
        
        # 简单的左右移动
        if current_x < target_x - 2:
            self.gc.release_key("LEFT")
            self.gc.press_key("RIGHT")
        elif current_x > target_x + 2:
            self.gc.release_key("RIGHT")
            self.gc.press_key("LEFT")
        else:
            self.gc.clear_movement()

        # 到达 JUMP 跳跃节点附近，顺畅起跳并切节点
        if action_type == "JUMP" and abs(current_x - target_x) <= 3:
            self.gc.tap_key(jump_key, 0.1)
            nodes = self.route_manager.nodes
            if nodes:
                self.route_manager.current_target_index = (self.route_manager.current_target_index + 1) % len(nodes)

    def _handle_climb_action(self, m_pos, player_state, config):
        """
        精进攀爬状态机：
        1. 支持读入 CLIMB_END (F4) 标记的登顶终点高度，达到高度立刻松开 UP 键；
        2. 若未标记 CLIMB_END，则使用纵向 Y 坐标不再改变 (0.35s) 的自动兜底判定。
        """
        jump_key = config.get("jump_key", "Alt")
        mx, my = m_pos
        now = time.time()
        nodes = self.route_manager.nodes
        n_count = len(nodes)
        curr_idx = self.route_manager.current_target_index
        target_node = self.route_manager.get_current_target_node()
        rx = target_node.x if target_node else mx

        # 查找是否有匹配的 CLIMB_END 登顶终点节点及其 Y 坐标与索引
        climb_end_y = None
        climb_end_idx = None
        if n_count > 0:
            for offset in range(1, n_count):
                idx = (curr_idx + offset) % n_count
                if nodes[idx].action_type == "CLIMB_END":
                    climb_end_y = nodes[idx].y
                    climb_end_idx = idx
                    break

        # 检查纵向 Y 坐标是否在向上改变，或 YOLO 判定为 climb 状态
        y_is_changing = (self.climb_last_y != -1 and my < self.climb_last_y) or (player_state == "climb")

        if y_is_changing or self.climb_caught:
            # 标记已进入成功挂绳攀爬阶段
            if not self.climb_caught:
                self.climb_caught = True
                print(f"[CLIMB] 检测到纵向坐标开始改变 (Y: {self.climb_last_y} -> {my})，抓绳成功！持续按上键...")
            
            # 释放左右方向键，持续按住上键
            self.gc.release_key("LEFT")
            self.gc.release_key("RIGHT")
            self.gc.press_key("UP")
            
            # 持续更新 Y 坐标与时间
            if my != self.climb_last_y:
                self.climb_last_y = my
                self.climb_last_y_time = now
            
            # 登顶判断 1：显式匹配到 CLIMB_END 高度 (my <= climb_end_y + 1)
            reached_by_marker = (climb_end_y is not None and my <= climb_end_y + 1)
            # 登顶判断 2：纵向 Y 坐标超过 0.35 秒不再改变 (兜底)
            reached_by_timeout = (now - self.climb_last_y_time > 0.35)

            if reached_by_marker or reached_by_timeout:
                reason = f"已到达 CLIMB_END 标记高度 (Y={climb_end_y})" if reached_by_marker else "纵向坐标稳定不再改变"
                print(f"[CLIMB] {reason}，攀爬完毕登顶平台！停止按上键，切换下一个节点")
                self.gc.release_key("UP")
                self.is_climbing_rope = False
                self.climb_caught = False
                self.climb_attempt = 0
                if nodes:
                    if climb_end_idx is not None:
                        # 切到 CLIMB_END 之后的节点
                        self.route_manager.current_target_index = (climb_end_idx + 1) % len(nodes)
                    else:
                        self.route_manager.current_target_index = (curr_idx + 1) % len(nodes)
        else:
            # 还未抓到绳子 (纵向坐标尚未改变)
            # 1. 检查横向偏差，若与绳子标记点距离较大 (偏差 > 1.5 像素)，先平滑移动对齐绳子正下方
            if abs(mx - rx) > 1.5:
                if mx < rx:
                    self.gc.release_key("LEFT")
                    self.gc.press_key("RIGHT")
                else:
                    self.gc.release_key("RIGHT")
                    self.gc.press_key("LEFT")
                return

            # 2. 已处于紧凑起跳点距离内 (|mx - rx| <= 1.5)，发起起跳
            if now - self.climb_start_time > 0.7:  # 每 0.7 秒起跳尝试一次
                self.climb_start_time = now
                self.climb_attempt += 1

                # 记录起跳前的 Y 坐标基准
                self.climb_last_y = my
                self.climb_last_y_time = now

                # 判定精确定位靠拢方向 (如果偏差 <= 0.5 则判定为正对绳子，垂直直跳)
                dir_key = None
                if mx < rx - 0.5:
                    dir_key = "RIGHT"
                    self.gc.release_key("LEFT")
                    self.gc.press_key("RIGHT")
                elif mx > rx + 0.5:
                    dir_key = "LEFT"
                    self.gc.release_key("RIGHT")
                    self.gc.press_key("LEFT")
                else:
                    self.gc.clear_movement()  # 垂直正对绳子，清除横向按键直跳

                # 按钮间隔 50ms 确保跳跃触发生效
                time.sleep(0.05)

                # 触发起跳
                self.gc.tap_key(jump_key)
                
                # 若有方向则进行微量弧线靠拢 (80ms)，直跳则直接按住 UP
                if dir_key:
                    time.sleep(0.08)
                    self.gc.press_key("UP")
                    time.sleep(0.08)
                    self.gc.release_key(dir_key)
                else:
                    time.sleep(0.05)
                    self.gc.press_key("UP")

                # 超时防卡死兜底：如果尝试 6 次都没挂上绳子
                if self.climb_attempt > 6:
                    print("【防卡死】多次起跳纵向坐标未改变，随机侧跳破局...")
                    self.gc.release_key("UP")
                    self.gc.press_key("RIGHT")
                    self.gc.tap_key(jump_key)
                    time.sleep(0.3)
                    self.gc.release_key("RIGHT")
                    
                    self.climb_attempt = 0
                    self.is_climbing_rope = False
                    self.climb_caught = False

    def _execute_combat_moving(self, p_pos, m_pos):
        if not p_pos or not m_pos:
            self.gc.clear_movement()
            return
        px = p_pos[0]
        py = p_pos[1]
        mx = m_pos[0]
        my = m_pos[1]
        atk_x = getattr(self, 'ATTACK_RANGE_X', 140)
        stop_dist = max(30, atk_x - 30)
        if px < mx - stop_dist:
            self.gc.release_key("LEFT")
            self.gc.press_key("RIGHT")
        elif px > mx + stop_dist:
            self.gc.release_key("RIGHT")
            self.gc.press_key("LEFT")
        else:
            self.gc.clear_movement()

    def _execute_combat(self, p_pos, same_level_monsters, config):
        self.gc.clear_movement()
        now = time.time()
        px = p_pos[0]
        py = p_pos[1]
        normal_atk_key = config.get("normal_atk_key", "C")
        normal_atk_range = config.get("normal_atk_range", 140)
        normal_atk_interval = config.get("normal_atk_interval", 0.6)
        aoe_skill_key = config.get("aoe_skill_key", "D")
        aoe_skill_range = config.get("aoe_skill_range", 200)
        aoe_skill_interval = config.get("aoe_skill_interval", 0.6)
        aoe_monster_count = config.get("aoe_monster_count", 3)
        aoe_dir_mode = config.get("aoe_dir_mode", "单向 (单侧面朝方向)")

        # same_level_monsters 已经过 update() 双重高度对齐筛选，直接作为有效怪物列表
        valid_monsters = same_level_monsters
        if not valid_monsters:
            return

        # 1. 优先判定群攻门槛 (按动态浮动的 aoe_skill_interval 冷却)
        is_single_dir = ("单向" in aoe_dir_mode)
        aoe_ready = (now - self.last_aoe_attack_time >= self.target_aoe_skill_interval)
        triggered_aoe = False

        if aoe_ready:
            if is_single_dir:
                # 单向判定：分别统计左右单侧在群攻范围内的怪物
                r_monsters = [m for m in valid_monsters if 0 <= (m[0] - px) <= aoe_skill_range]
                l_monsters = [m for m in valid_monsters if 0 <= (px - m[0]) <= aoe_skill_range]
                if len(r_monsters) >= aoe_monster_count:
                    # 100% 确保面向右侧怪群：轻点 RIGHT，再施放群攻
                    self.gc.tap_key("RIGHT", 0.03)
                    self.gc.tap_key(aoe_skill_key)
                    self.last_aoe_attack_time = now
                    # 重新生成下次群攻浮动 CD (±10%)
                    if hasattr(self.gc, 'jitter') and self.gc.jitter:
                        self.target_aoe_skill_interval = self.gc.jitter.calc_floating_interval(aoe_skill_interval, ratio=0.10)
                    else:
                        self.target_aoe_skill_interval = aoe_skill_interval
                    triggered_aoe = True
                    print(f"[COMBAT] 【单向群攻触发(右侧)】怪数={len(r_monsters)} >= {aoe_monster_count}，转向 [RIGHT] 施放群攻 [{aoe_skill_key}]")
                elif len(l_monsters) >= aoe_monster_count:
                    # 100% 确保面向左侧怪群：轻点 LEFT，再施放群攻
                    self.gc.tap_key("LEFT", 0.03)
                    self.gc.tap_key(aoe_skill_key)
                    self.last_aoe_attack_time = now
                    if hasattr(self.gc, 'jitter') and self.gc.jitter:
                        self.target_aoe_skill_interval = self.gc.jitter.calc_floating_interval(aoe_skill_interval, ratio=0.10)
                    else:
                        self.target_aoe_skill_interval = aoe_skill_interval
                    triggered_aoe = True
                    print(f"[COMBAT] 【单向群攻触发(左侧)】怪数={len(l_monsters)} >= {aoe_monster_count}，转向 [LEFT] 施放群攻 [{aoe_skill_key}]")
            else:
                # 双向判定：以玩家为中心左右两侧总数
                aoe_list = [m for m in valid_monsters if abs(m[0] - px) <= aoe_skill_range]
                if len(aoe_list) >= aoe_monster_count:
                    r_cnt = sum(1 for m in aoe_list if m[0] >= px)
                    l_cnt = len(aoe_list) - r_cnt
                    target_dir = "RIGHT" if r_cnt >= l_cnt else "LEFT"
                    self.gc.tap_key(target_dir, 0.03)
                    self.gc.tap_key(aoe_skill_key)
                    self.last_aoe_attack_time = now
                    if hasattr(self.gc, 'jitter') and self.gc.jitter:
                        self.target_aoe_skill_interval = self.gc.jitter.calc_floating_interval(aoe_skill_interval, ratio=0.10)
                    else:
                        self.target_aoe_skill_interval = aoe_skill_interval
                    triggered_aoe = True
                    print(f"[COMBAT] 【双向群攻触发】总怪数={len(aoe_list)} >= {aoe_monster_count}，转向 [{target_dir}] 施放群攻 [{aoe_skill_key}]")

        # 2. 若未触发群攻，判定普攻 (按动态浮动的 normal_atk_interval 冷却)
        if not triggered_aoe:
            normal_ready = (now - self.last_normal_attack_time >= self.target_normal_atk_interval)
            if normal_ready:
                normal_list = [m for m in valid_monsters if abs(m[0] - px) <= normal_atk_range]
                if normal_list:
                    target_m = min(normal_list, key=lambda m: abs(m[0] - px))
                    # 100% 确保面向怪物：点按对应方向键再出招
                    target_dir = "RIGHT" if target_m[0] >= px else "LEFT"
                    self.gc.tap_key(target_dir, 0.03)
                    self.gc.tap_key(normal_atk_key)
                    self.last_normal_attack_time = now
                    # 重新生成下次普攻浮动 CD (±10%)
                    if hasattr(self.gc, 'jitter') and self.gc.jitter:
                        self.target_normal_atk_interval = self.gc.jitter.calc_floating_interval(normal_atk_interval, ratio=0.10)
                    else:
                        self.target_normal_atk_interval = normal_atk_interval
                    print(f"[COMBAT] 【普攻触发】目标距X={abs(target_m[0]-px)}px，转向 [{target_dir}] 施放普攻 [{normal_atk_key}]")

    def reset(self):
        self._set_state(FSMState.IDLE, force=True)
        self.first_monster_seen_time = 0
        self.no_combat_start_time = 0
        self.micro_idle_end_time = 0
        self.is_escaping_platform_danger = False
        self.current_platform_bounds = None
        self.gc.release_all_keys()
