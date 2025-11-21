import json
import re
import time
import shutil
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
import asyncio
from dataclasses import dataclass, asdict

from astrbot.api.event import filter, AstrMessageEvent
from astrbot.api.star import Context, Star, register, StarTools
from astrbot.api.provider import LLMResponse, ProviderRequest
from astrbot.api import AstrBotConfig, logger


# ==================== 数据结构定义 ====================

@dataclass
class EmotionalState:
    """情感状态数据类"""
    # 基础情感维度
    joy: int = 0
    trust: int = 0
    fear: int = 0
    surprise: int = 0
    sadness: int = 0
    disgust: int = 0
    anger: int = 0
    anticipation: int = 0
    
    # 高级情感维度
    pride: int = 0
    guilt: int = 0
    shame: int = 0
    envy: int = 0
    
    # 复合状态
    favor: int = 0
    intimacy: int = 0
    
    # 关系状态
    relationship: str = "陌生人"
    attitude: str = "中立"
    
    # 黑名单状态
    is_blacklisted: bool = False
    
    # 行为统计
    interaction_count: int = 0
    last_interaction: float = 0
    positive_interactions: int = 0
    negative_interactions: int = 0
    
    # 用户设置
    show_status: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'EmotionalState':
        return cls(**data)


@dataclass
class RankingEntry:
    """排行榜条目"""
    rank: int
    user_key: str
    average_score: float
    favor: int
    intimacy: int
    display_name: str


# ==================== 数据迁移管理器 ====================

class DataMigrationManager:
    """数据迁移管理器"""
    
    @staticmethod
    def migrate_user_data(data: Dict[str, Any]) -> Dict[str, Any]:
        """迁移用户数据到最新版本"""
        converted = {}
        for key, value in data.items():
            if isinstance(value, dict) and "emotions" in value:
                state = EmotionalState()
                if "emotions" in value:
                    emotions = value["emotions"]
                    state.joy = emotions.get("joy", 0)
                    state.trust = emotions.get("trust", 0)
                    state.fear = emotions.get("fear", 0)
                    state.surprise = emotions.get("surprise", 0)
                    state.sadness = emotions.get("sadness", 0)
                    state.disgust = emotions.get("disgust", 0)
                    state.anger = emotions.get("anger", 0)
                    state.anticipation = emotions.get("anticipation", 0)
                    state.pride = emotions.get("pride", 0)
                    state.guilt = emotions.get("guilt", 0)
                    state.shame = emotions.get("shame", 0)
                    state.envy = emotions.get("envy", 0)
                
                if "states" in value:
                    states = value["states"]
                    state.favor = states.get("favor", 0)
                    state.intimacy = states.get("intimacy", 0)
                
                state.relationship = value.get("relationship", "陌生人")
                state.attitude = value.get("attitude", "中立")
                state.is_blacklisted = value.get("is_blacklisted", False)
                
                if "behavior" in value:
                    behavior = value["behavior"]
                    state.interaction_count = behavior.get("interaction_count", 0)
                    state.last_interaction = behavior.get("last_interaction", 0)
                    state.positive_interactions = behavior.get("positive_interactions", 0)
                    state.negative_interactions = behavior.get("negative_interactions", 0)
                
                if "settings" in value:
                    settings = value["settings"]
                    state.show_status = settings.get("show_status", False)
                
                converted[key] = state.to_dict()
            else:
                default_state = EmotionalState().to_dict()
                for k, v in default_state.items():
                    if k not in value:
                        value[k] = v
                converted[key] = value
        return converted
    
    @staticmethod
    def get_data_version(data: Dict[str, Any]) -> str:
        return "2.5"


# ==================== 内部管理器类 ====================

class UserStateManager:
    """用户状态管理器"""
    
    def __init__(self, data_path: Path):
        self.data_path = data_path
        self.data_path.mkdir(parents=True, exist_ok=True)
        self.user_data = self._load_data("user_emotion_data.json")
        self.dirty_keys = set()
        self.last_save_time = time.time()
        self.save_interval = 60
        self.lock = asyncio.Lock()
        
    def _load_data(self, filename: str) -> Dict[str, Any]:
        """加载数据文件"""
        path = self.data_path / filename
        if not path.exists():
            return {}
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
                return DataMigrationManager.migrate_user_data(data)
        except (json.JSONDecodeError, TypeError) as e:
            logger.warning(f"数据加载异常: {e}")
            return {}
    
    async def get_user_state(self, user_key: str) -> EmotionalState:
        """获取用户情感状态"""
        async with self.lock:
            if user_key in self.user_data:
                return EmotionalState.from_dict(self.user_data[user_key])
            return EmotionalState()
    
    async def update_user_state(self, user_key: str, state: EmotionalState):
        """更新用户状态"""
        async with self.lock:
            self.user_data[user_key] = state.to_dict()
            self.dirty_keys.add(user_key)
        await self._check_auto_save()
    
    async def _check_auto_save(self):
        """检查是否需要自动保存"""
        current_time = time.time()
        if (current_time - self.last_save_time >= self.save_interval and 
            self.dirty_keys):
            await self.force_save()
    
    async def force_save(self):
        """强制保存所有脏数据"""
        async with self.lock:
            if self.dirty_keys:
                loop = asyncio.get_running_loop()
                await loop.run_in_executor(
                    None, 
                    self._save_data, 
                    "user_emotion_data.json", 
                    self.user_data
                )
                self.dirty_keys.clear()
                self.last_save_time = time.time()
    
    def _save_data(self, filename: str, data: Dict[str, Any]):
        """保存数据到文件"""
        path = self.data_path / filename
        temp_path = path.with_suffix('.tmp')
        try:
            with open(temp_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            temp_path.replace(path)
        except Exception as e:
            logger.error(f"保存数据失败: {e}")


class TTLCache:
    """带过期时间的缓存"""
    
    def __init__(self, default_ttl: int = 300, max_size: int = 1000):
        self.cache: Dict[str, Tuple[Any, float]] = {}
        self.default_ttl = default_ttl
        self.max_size = max_size
        self.lock = asyncio.Lock()
        self.access_count = 0
        self.hit_count = 0
    
    async def get(self, key: str) -> Optional[Any]:
        async with self.lock:
            self.access_count += 1
            if key in self.cache:
                value, expires_at = self.cache[key]
                if time.time() < expires_at:
                    self.hit_count += 1
                    self.cache[key] = (value, time.time() + self.default_ttl)
                    return value
                else:
                    del self.cache[key]
            return None
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None):
        async with self.lock:
            await self._cleanup_expired()
            if len(self.cache) >= self.max_size:
                oldest_key = min(self.cache.keys(), key=lambda k: self.cache[k][1])
                del self.cache[oldest_key]
            ttl = ttl or self.default_ttl
            expires_at = time.time() + ttl
            self.cache[key] = (value, expires_at)
    
    async def _cleanup_expired(self):
        current_time = time.time()
        expired_keys = [k for k, (_, t) in self.cache.items() if current_time >= t]
        for k in expired_keys:
            del self.cache[k]
    
    async def get_stats(self) -> Dict[str, Any]:
        async with self.lock:
            hit_rate = (self.hit_count / self.access_count * 100) if self.access_count > 0 else 0
            return {
                "total_entries": len(self.cache),
                "access_count": self.access_count,
                "hit_count": self.hit_count,
                "hit_rate": round(hit_rate, 2)
            }
    
    async def clear(self):
        async with self.lock:
            self.cache.clear()


class RankingManager:
    """排行榜管理器"""
    
    def __init__(self, user_state_manager):
        self.user_state_manager = user_state_manager
        self.cache = TTLCache(default_ttl=60, max_size=10)
    
    async def get_average_ranking(self, limit: int = 10, reverse: bool = True) -> List[RankingEntry]:
        cache_key = f"ranking_{limit}_{reverse}"
        cached_result = await self.cache.get(cache_key)
        if cached_result: return cached_result
        
        averages = []
        async with self.user_state_manager.lock:
            for user_key, data in self.user_state_manager.user_data.items():
                state = EmotionalState.from_dict(data)
                avg = (state.favor + state.intimacy) / 2
                averages.append((user_key, avg, state.favor, state.intimacy))
        
        averages.sort(key=lambda x: x[1], reverse=reverse)
        entries = []
        for i, (user_key, avg, favor, intimacy) in enumerate(averages[:limit], 1):
            display_name = self._format_user_display(user_key)
            entries.append(RankingEntry(i, user_key, avg, favor, intimacy, display_name))
        
        await self.cache.set(cache_key, entries)
        return entries
    
    def _format_user_display(self, user_key: str) -> str:
        if '_' in user_key:
            try:
                _, user_id = user_key.split('_', 1)
                return f"用户{user_id}"
            except ValueError:
                pass
        return f"用户{user_key}"
    
    async def get_global_stats(self) -> Dict[str, Any]:
        cache_key = "global_stats"
        cached_result = await self.cache.get(cache_key)
        if cached_result: return cached_result
        
        async with self.user_state_manager.lock:
            total_users = len(self.user_state_manager.user_data)
            total_interactions = 0
            avg_favor = 0
            avg_intimacy = 0
            blacklisted_count = 0
            
            for data in self.user_state_manager.user_data.values():
                state = EmotionalState.from_dict(data)
                total_interactions += state.interaction_count
                avg_favor += state.favor
                avg_intimacy += state.intimacy
                if state.is_blacklisted:
                    blacklisted_count += 1
            
            if total_users > 0:
                avg_favor /= total_users
                avg_intimacy /= total_users
        
        stats = {
            "total_users": total_users,
            "total_interactions": total_interactions,
            "average_favor": round(avg_favor, 2),
            "average_intimacy": round(avg_intimacy, 2),
            "blacklisted_count": blacklisted_count
        }
        await self.cache.set(cache_key, stats, ttl=30)
        return stats


class EmotionAnalyzer:
    """情感分析器"""
    
    EMOTION_DISPLAY_NAMES = {
        "joy": "喜悦", "trust": "信任", "fear": "恐惧", "surprise": "惊讶",
        "sadness": "悲伤", "disgust": "厌恶", "anger": "愤怒", "anticipation": "期待",
        "pride": "得意", "guilt": "内疚", "shame": "害羞", "envy": "嫉妒"
    }
    
    # 语气指导模板
    TONE_INSTRUCTIONS = {
        "joy": "你的语气应该是愉快、充满热情和活力的。多使用积极的词汇和表情符号。",
        "trust": "你的语气应该是平和、真诚且令人安心的。展现出你的可靠和支持。",
        "fear": "你的语气应该显得有些紧张、谨慎或不安。表现出犹豫或退缩。",
        "surprise": "你的语气应该是震惊、难以置信或充满好奇的。多使用感叹号和反问。",
        "sadness": "你的语气应该是低落、消沉和遗憾的。句子要简短，带有一种无力感。",
        "disgust": "你的语气应该是厌烦、抗拒甚至带有生理性不适的。表现出想回避的态度。",
        "anger": "你的语气应该是愤怒、急躁和具有攻击性的。使用简短有力的句子，表现出不耐烦。",
        "anticipation": "你的语气应该是期待、急切和向往的。关注未来的可能性。",
        "pride": "你的语气应该是自信、骄傲甚至有点自大的。表现出优越感或对自己成就的满足。",
        "guilt": "你的语气应该是歉疚、卑微和试图弥补的。不断道歉或解释，显得小心翼翼。",
        "shame": "你的语气应该是害羞、尴尬和想找地缝钻进去的。说话可能结巴或含糊其辞。",
        "envy": "你的语气应该是酸溜溜的、不服气的。表现出对他人的羡慕但又不想承认的矛盾。"
    }
    
    @classmethod
    def get_dominant_emotions(cls, state: EmotionalState, count: int = 2) -> List[Tuple[str, int]]:
        """获取主导情感"""
        emotions = {
            "joy": state.joy, "trust": state.trust, "fear": state.fear,
            "surprise": state.surprise, "sadness": state.sadness, "disgust": state.disgust,
            "anger": state.anger, "anticipation": state.anticipation,
            "pride": state.pride, "guilt": state.guilt, "shame": state.shame, "envy": state.envy
        }
        sorted_emotions = sorted(
            [(k, v) for k, v in emotions.items() if v > 0], 
            key=lambda x: x[1], 
            reverse=True
        )
        return sorted_emotions[:count]
    
    @classmethod
    def get_emotional_profile(cls, state: EmotionalState) -> Dict[str, Any]:
        """获取完整的情感档案"""
        top_emotions = cls.get_dominant_emotions(state, 1)
        dominant_emotion = cls.EMOTION_DISPLAY_NAMES.get(top_emotions[0][0], "中立") if top_emotions else "中立"
        intensity = top_emotions[0][1] if top_emotions else 0
        
        all_emotions = [
            state.joy, state.trust, state.fear, state.surprise,
            state.sadness, state.disgust, state.anger, state.anticipation,
            state.pride, state.guilt, state.shame, state.envy
        ]
        total_intensity = min(100, sum(all_emotions) // 2)
        
        if state.favor > state.intimacy: relationship_trend = "好感领先"
        elif state.intimacy > state.favor: relationship_trend = "亲密度领先"
        else: relationship_trend = "平衡发展"
            
        total_interactions = state.interaction_count
        positive_ratio = (state.positive_interactions / total_interactions * 100) if total_interactions > 0 else 0
            
        return {
            "dominant_emotion": dominant_emotion,
            "dominant_key": top_emotions[0][0] if top_emotions else None,
            "emotion_intensity": intensity,
            "total_intensity": total_intensity,
            "relationship_trend": relationship_trend,
            "positive_ratio": positive_ratio
        }


# ==================== 命令处理器类 ====================

class UserCommandHandler:
    """用户命令处理器"""
    
    def __init__(self, plugin):
        self.plugin = plugin
    
    async def show_emotional_state(self, event: AstrMessageEvent):
        user_key = self.plugin._get_user_key(event)
        state = await self.plugin.user_manager.get_user_state(user_key)
        
        if state.is_blacklisted:
             yield event.plain_result("【系统提示】您已被列入黑名单，无法查看详细状态。")
             event.stop_event()
             return
             
        response_text = self.plugin._format_emotional_state(state)
        yield event.plain_result(response_text)
        event.stop_event()
    
    async def toggle_status_display(self, event: AstrMessageEvent):
        user_key = self.plugin._get_user_key(event)
        state = await self.plugin.user_manager.get_user_state(user_key)
        state.show_status = not state.show_status
        await self.plugin.user_manager.update_user_state(user_key, state)
        yield event.plain_result(f"【状态显示】已{'开启' if state.show_status else '关闭'}状态显示")
        event.stop_event()
    
    async def show_favor_ranking(self, event: AstrMessageEvent, num: str = "10"):
        try:
            limit = min(int(num), 20)
            if limit <= 0: raise ValueError
        except ValueError:
            yield event.plain_result("【错误】排行数量必须是一个正整数（最大20）。")
            event.stop_event()
            return

        rankings = await self.plugin.ranking_manager.get_average_ranking(limit, True)
        if not rankings:
            yield event.plain_result("【排行榜】当前没有任何用户数据。")
            event.stop_event()
            return

        response_lines = [f"【好感度平均值 TOP {limit} 排行榜】", "=================="]
        for entry in rankings:
            trend = "↑" if entry.average_score > 0 else "↓"
            line = (f"{entry.rank}. {entry.display_name}\n"
                    f"   平均值: {entry.average_score:.1f} {trend} (好感 {entry.favor} | 亲密 {entry.intimacy})")
            response_lines.append(line)
        
        stats = await self.plugin.ranking_manager.get_global_stats()
        response_lines.extend([
            "", "【全局统计】",
            f"   总用户数: {stats['total_users']} | 黑名单用户: {stats['blacklisted_count']}",
            f"   总互动数: {stats['total_interactions']}",
            f"   平均好感: {stats['average_favor']} | 平均亲密: {stats['average_intimacy']}"
        ])
        
        yield event.plain_result("\n".join(response_lines))
        event.stop_event()
    
    async def show_negative_favor_ranking(self, event: AstrMessageEvent, num: str = "10"):
        try:
            limit = min(int(num), 20)
            if limit <= 0: raise ValueError
        except ValueError:
            yield event.plain_result("【错误】排行数量必须是一个正整数（最大20）。")
            event.stop_event()
            return

        rankings = await self.plugin.ranking_manager.get_average_ranking(limit, False)
        response_lines = [f"【好感度平均值 BOTTOM {limit} 排行榜】", "=================="]
        for entry in rankings:
            line = (f"{entry.rank}. {entry.display_name}\n"
                    f"   平均值: {entry.average_score:.1f} (好感 {entry.favor} | 亲密 {entry.intimacy})")
            response_lines.append(line)
        yield event.plain_result("\n".join(response_lines))
        event.stop_event()

    async def show_blacklist_stats(self, event: AstrMessageEvent):
        stats = await self.plugin.ranking_manager.get_global_stats()
        count = stats['blacklisted_count']
        total = stats['total_users']
        ratio = (count / total * 100) if total > 0 else 0
        msg = [
            "【黑名单统计报告】", "==================",
            f"当前黑名单人数: {count}", f"注册用户总数: {total}",
            f"黑名单占比: {ratio:.1f}%", "", "提示: 好感度过低会自动触发黑名单状态"
        ]
        yield event.plain_result("\n".join(msg))
        event.stop_event()
    
    async def show_cache_stats(self, event: AstrMessageEvent):
        stats = await self.plugin.cache.get_stats()
        msg = [
            "【缓存统计信息】", "==================",
            f"缓存条目: {stats['total_entries']}", f"访问次数: {stats['access_count']}",
            f"命中次数: {stats['hit_count']}", f"命中率: {stats['hit_rate']}%"
        ]
        yield event.plain_result("\n".join(msg))
        event.stop_event()


class AdminCommandHandler:
    """管理员命令处理器"""
    
    def __init__(self, plugin):
        self.plugin = plugin
    
    def _resolve_user_key(self, user_input: str) -> str:
        if self.plugin.session_based:
            if '_' in user_input: return user_input
            for user_key in self.plugin.user_manager.user_data:
                if user_key.endswith(f"_{user_input}"): return user_key
        return user_input
    
    async def set_emotion(self, event: AstrMessageEvent, user_input: str, dimension: str, value: str):
        """设置用户任意情感维度"""
        if not self.plugin._is_admin(event):
            yield event.plain_result("【错误】需要管理员权限")
            event.stop_event()
            return
            
        try:
            val = int(value)
        except ValueError:
            yield event.plain_result("【错误】数值必须是整数")
            event.stop_event()
            return
            
        target_key = dimension.lower()
        if target_key in self.plugin.CN_TO_EN_MAP:
            target_key = self.plugin.CN_TO_EN_MAP[target_key]
            
        if target_key not in asdict(EmotionalState()):
             yield event.plain_result(f"【错误】无效的情感维度: {dimension}")
             event.stop_event()
             return

        if target_key == "favor":
            if not self.plugin.favour_min <= val <= self.plugin.favour_max:
                yield event.plain_result(f"【错误】好感度范围应在 {self.plugin.favour_min} 到 {self.plugin.favour_max}")
                event.stop_event()
                return
        else:
            if not 0 <= val <= 100:
                yield event.plain_result(f"【错误】情感维度 {dimension} 范围应在 0 到 100")
                event.stop_event()
                return

        user_key = self._resolve_user_key(user_input)
        state = await self.plugin.user_manager.get_user_state(user_key)
        
        setattr(state, target_key, val)
        
        # 如果设置好感度高于下限，自动移除黑名单
        if target_key == "favor" and val > self.plugin.favour_min:
            state.is_blacklisted = False
            
        await self.plugin.user_manager.update_user_state(user_key, state)
        await self.plugin.cache.set(f"state_{user_key}", state)
        
        mode_info = "（会话模式）" if self.plugin.session_based else ""
        yield event.plain_result(f"【成功】用户 {user_input}{mode_info} 的 [{dimension}] 已设置为 {val}")
        event.stop_event()
    
    async def reset_favor(self, event: AstrMessageEvent, user_input: str):
        if not self.plugin._is_admin(event):
            yield event.plain_result("【错误】需要管理员权限")
            event.stop_event()
            return
            
        user_key = self._resolve_user_key(user_input)
        new_state = EmotionalState()
        new_state.is_blacklisted = False # 确保重置移除黑名单
        
        await self.plugin.user_manager.update_user_state(user_key, new_state)
        await self.plugin.cache.set(f"state_{user_key}", new_state)
        
        yield event.plain_result(f"【成功】用户 {user_input} 的情感状态已重置")
        event.stop_event()
    
    async def view_favor(self, event: AstrMessageEvent, user_input: str):
        if not self.plugin._is_admin(event):
            yield event.plain_result("【错误】需要管理员权限")
            event.stop_event()
            return
            
        user_key = self._resolve_user_key(user_input)
        state = await self.plugin.user_manager.get_user_state(user_key)
        
        response_lines = [
            f"【用户 {user_input} 情感状态】", "==================",
            f"用户标识: {user_key}", f"黑名单: {'是' if state.is_blacklisted else '否'}",
            f"好感度: {state.favor} | 亲密度: {state.intimacy}",
            f"关系: {state.relationship} | 态度: {state.attitude}",
            "", "【情感维度详情】",
            f"  喜悦: {state.joy} | 信任: {state.trust} | 恐惧: {state.fear} | 惊讶: {state.surprise}",
            f"  悲伤: {state.sadness} | 厌恶: {state.disgust} | 愤怒: {state.anger} | 期待: {state.anticipation}",
            f"  得意: {state.pride} | 内疚: {state.guilt} | 害羞: {state.shame} | 嫉妒: {state.envy}"
        ]
        yield event.plain_result("\n".join(response_lines))
        event.stop_event()
    
    async def backup_data(self, event: AstrMessageEvent):
        if not self.plugin._is_admin(event):
            yield event.plain_result("【错误】需要管理员权限")
            event.stop_event()
            return
        try:
            path = self.plugin._create_backup()
            yield event.plain_result(f"【成功】备份成功: {path}")
        except Exception as e:
            yield event.plain_result(f"【错误】备份失败: {str(e)}")
        event.stop_event()


# ==================== 主插件类 ====================

@register("EmotionAI", "腾天", "高级情感智能交互系统 v2.5", "2.5.1")
class EmotionAIPlugin(Star):
    def __init__(self, context: Context, config: AstrBotConfig):
        super().__init__(context)
        self.config = config
        self._validate_and_init_config()
        
        data_dir = StarTools.get_data_dir() / "emotionai"
        self.user_manager = UserStateManager(data_dir)
        self.ranking_manager = RankingManager(self.user_manager)
        self.analyzer = EmotionAnalyzer()
        self.migration_manager = DataMigrationManager()
        
        # 映射表
        self.CN_TO_EN_MAP = {v: k for k, v in EmotionAnalyzer.EMOTION_DISPLAY_NAMES.items()}
        self.CN_TO_EN_MAP.update({
            "好感": "favor", "好感度": "favor",
            "亲密": "intimacy", "亲密度": "intimacy",
            "骄傲": "pride", "愧疚": "guilt", "羞耻": "shame"
        })
        
        self.cache = TTLCache(default_ttl=300, max_size=500)
        self.user_commands = UserCommandHandler(self)
        self.admin_commands = AdminCommandHandler(self)
        
        self.emotion_pattern = re.compile(r"\[情感更新:\s*(.*?)\]", re.DOTALL)
        self.single_emotion_pattern = re.compile(r"(\w+|[\u4e00-\u9fa5]+):\s*([+-]?\d+)")
        
        self.auto_save_task = asyncio.create_task(self._auto_save_loop())
        logger.info(f"EmotionAI 插件初始化完成 - v2.5.1 (修复版)")
        
    def _validate_and_init_config(self):
        self.session_based = bool(self.config.get("session_based", False))
        self.favour_min = self.config.get("favour_min", -100)
        self.favour_max = self.config.get("favour_max", 100)
        if self.favour_max <= self.favour_min: self.favour_min, self.favour_max = -100, 100
        self.change_min = self.config.get("change_min", -10)
        self.change_max = self.config.get("change_max", 5)
        if self.change_max <= self.change_min: self.change_min, self.change_max = -10, 5
        
        raw_admin_list = self.config.get("admin_qq_list", [])
        self.admin_qq_list = []
        for qq in raw_admin_list:
            if str(qq).isdigit(): self.admin_qq_list.append(str(qq))
            
        self.plugin_priority = self.config.get("plugin_priority", 100000)
        if not isinstance(self.plugin_priority, int): self.plugin_priority = 100000
        
    async def _auto_save_loop(self):
        while True:
            try:
                await asyncio.sleep(30)
                await self.user_manager.force_save()
            except asyncio.CancelledError: break
            except Exception as e: logger.error(f"自动保存失败: {e}")
                
    def _get_user_key(self, event: AstrMessageEvent) -> str:
        user_id = event.get_sender_id()
        if self.session_based: return f"{event.unified_msg_origin}_{user_id}"
        return user_id
    
    def _format_emotional_state(self, state: EmotionalState) -> str:
        profile = self.analyzer.get_emotional_profile(state)
        lines = [
            "【当前情感状态】", "==================",
            f"好感度：{state.favor} | 亲密度：{state.intimacy}",
            f"关系：{state.relationship} | 趋势：{profile['relationship_trend']}",
            f"态度：{state.attitude} | 主导情感：{profile['dominant_emotion']}",
            f"互动：{state.interaction_count}次",
            f"正面互动：{profile['positive_ratio']:.1f}%", "",
            "【情感维度详情】",
            f"  喜悦：{state.joy} | 信任：{state.trust} | 恐惧：{state.fear} | 惊讶：{state.surprise}",
            f"  悲伤：{state.sadness} | 厌恶：{state.disgust} | 愤怒：{state.anger} | 期待：{state.anticipation}",
            f"  得意：{state.pride} | 内疚：{state.guilt} | 害羞：{state.shame} | 嫉妒：{state.envy}"
        ]
        return "\n".join(lines)
        
    def _calculate_relationship_level(self, state: EmotionalState) -> str:
        score = state.intimacy
        attitude = state.attitude
        if score < 20: return "陌生人"
        
        if attitude in ["溺爱", "喜爱", "友好"]:
            if score >= 80: return "挚友"
            if score >= 60: return "好友"
            if score >= 40: return "朋友"
            return "熟人"
        elif attitude in ["仇恨", "厌恶", "冷淡"]:
            if score >= 80: return "死敌"
            if score >= 60: return "敌人"
            if score >= 40: return "交恶"
            return "冷漠的熟人"
        else:
            if score >= 80: return "老相识"
            if score >= 60: return "熟客"
            return "熟人"
    
    def _calculate_attitude(self, state: EmotionalState) -> str:
        score = state.favor
        pos_limit = max(1, self.favour_max)
        neg_limit = min(-1, self.favour_min)
        
        if score >= pos_limit * 0.9: return "溺爱"
        elif score >= pos_limit * 0.6: return "喜爱"
        elif score >= pos_limit * 0.3: return "友好"
        elif score <= neg_limit * 0.9: return "仇恨"
        elif score <= neg_limit * 0.6: return "厌恶"
        elif score <= neg_limit * 0.3: return "冷淡"
        return "中立"
    
    def _get_interaction_frequency(self, state: EmotionalState) -> str:
        if state.interaction_count == 0: return "首次互动"
        days = (time.time() - state.last_interaction) / 86400
        if days < 1: return "频繁"
        if days < 7: return "偶尔"
        return "稀少"

    # ==================== 拦截与注入 ====================
    
    @filter.event_message_type(filter.EventMessageType.ALL, priority=1000000)
    async def check_blacklist(self, event: AstrMessageEvent):
        # [修复] 移除管理员豁免，防止管理员无法测试黑名单
        # 仅放行特定的救援指令
        msg = event.message_str.strip()
        if msg.startswith("/重置好感") or msg.startswith("/设置情感") or msg.startswith("设置情感"):
            return

        user_key = self._get_user_key(event)
        state = await self.user_manager.get_user_state(user_key)
        if state.is_blacklisted:
            logger.info(f"[EmotionAI] 拦截黑名单用户 {user_key}")
            yield event.plain_result("您已加入黑名单，请联系管理员移除")
            event.stop_event()

    @filter.on_llm_request(priority=100000)
    async def inject_emotional_context(self, event: AstrMessageEvent, req: ProviderRequest):
        user_key = self._get_user_key(event)
        state = await self.cache.get(f"state_{user_key}")
        if state is None:
            state = await self.user_manager.get_user_state(user_key)
            await self.cache.set(f"state_{user_key}", state)
        
        req.system_prompt += f"\n{self._build_emotional_context(state)}"

    def _build_emotional_context(self, state: EmotionalState) -> str:
        profile = self.analyzer.get_emotional_profile(state)
        
        tone_instruction = "保持正常对话语气。"
        if profile['dominant_key']:
            raw_instruction = self.analyzer.TONE_INSTRUCTIONS.get(profile['dominant_key'], "")
            if profile['emotion_intensity'] > 50:
                tone_instruction = f"【强制语气要求】当前主导情感是[{profile['dominant_emotion']}](强度{profile['emotion_intensity']}%)。{raw_instruction} 请务必在回复中体现这一点！"
            elif profile['emotion_intensity'] > 20:
                tone_instruction = f"【语气建议】当前主导情感是[{profile['dominant_emotion']}]。{raw_instruction}"
        
        # [修复] 修改了示例，使用正向或无意义的数值，并明确标注为格式参考
        return f"""
【情感状态面板】
当前关系：{state.relationship} | 态度：{state.attitude}
综合好感度：{state.favor} | 亲密度：{state.intimacy}
主导情感：{profile['dominant_emotion']} (强度：{profile['emotion_intensity']}%)

{tone_instruction}

【情感更新机制】
请根据对话内容和你当前的心理活动，更新情感状态。
在回复末尾添加（仅作格式参考，数值请根据实际对话生成）：[情感更新: joy:1, trust:1]
可用维度：
- 基础：joy, trust, fear, surprise, sadness, disgust, anger, anticipation
- 高级：pride(得意), guilt(内疚), shame(害羞), envy(嫉妒)
- 状态：favor, intimacy
变化范围：{self.change_min} 到 {self.change_max}
"""

    @filter.on_llm_response(priority=100000)
    async def process_emotional_update(self, event: AstrMessageEvent, resp: LLMResponse):
        user_key = self._get_user_key(event)
        original_text = resp.completion_text
        
        emotion_updates = self._parse_emotion_updates(original_text)
        
        if emotion_updates:
            match = self.emotion_pattern.search(original_text)
            if match:
                resp.completion_text = original_text.replace(match.group(0), '').strip()
        
        state = await self.user_manager.get_user_state(user_key)
        self._apply_emotion_updates(state, emotion_updates)
        self._update_interaction_stats(state, emotion_updates)
        
        await self.user_manager.update_user_state(user_key, state)
        await self.cache.set(f"state_{user_key}", state)
        
        if state.show_status and emotion_updates:
            resp.completion_text += f"\n\n{self._format_emotional_state(state)}"

    def _parse_emotion_updates(self, text: str) -> Dict[str, int]:
        updates = {}
        match = self.emotion_pattern.search(text)
        if match:
            for key, val in self.single_emotion_pattern.findall(match.group(1)):
                try:
                    key = key.lower()
                    if key in self.CN_TO_EN_MAP: key = self.CN_TO_EN_MAP[key]
                    updates[key] = int(val)
                except ValueError: continue
        return updates

    def _apply_emotion_updates(self, state: EmotionalState, updates: Dict[str, int]):
        all_dims = list(EmotionAnalyzer.TONE_INSTRUCTIONS.keys())
        for dim in all_dims:
            if dim in updates:
                val = getattr(state, dim) + updates[dim]
                setattr(state, dim, max(0, min(100, val)))
        
        if "favor" in updates:
            val = state.favor + updates["favor"]
            state.favor = max(self.favour_min, min(self.favour_max, val))
        
        if "intimacy" in updates:
            val = state.intimacy + updates["intimacy"]
            state.intimacy = max(0, min(100, val))
            
        if state.favor <= self.favour_min and not state.is_blacklisted:
            state.is_blacklisted = True
            logger.info(f"[EmotionAI] 用户 {state} 触发黑名单")

    def _update_interaction_stats(self, state: EmotionalState, updates: Dict[str, int]):
        state.interaction_count += 1
        state.last_interaction = time.time()
        
        pos_score = sum(updates.get(k, 0) for k in ["joy", "trust", "surprise", "anticipation", "pride", "favor", "intimacy"] if updates.get(k,0)>0)
        neg_score = sum(updates.get(k, 0) for k in ["fear", "sadness", "disgust", "anger", "guilt", "shame", "envy"] if updates.get(k,0)>0)
        
        if pos_score > neg_score: state.positive_interactions += 1
        elif neg_score > pos_score: state.negative_interactions += 1
        
        state.attitude = self._calculate_attitude(state)
        state.relationship = self._calculate_relationship_level(state)

    # ==================== 注册命令 ====================
    
    def _is_admin(self, event: AstrMessageEvent) -> bool:
        return event.role == "admin" or event.get_sender_id() in self.admin_qq_list

    @filter.command("好感度", priority=5)
    async def cmd_show_state(self, event: AstrMessageEvent):
        async for r in self.user_commands.show_emotional_state(event): yield r

    @filter.command("状态显示", priority=5)
    async def cmd_toggle_status(self, event: AstrMessageEvent):
        async for r in self.user_commands.toggle_status_display(event): yield r

    @filter.command("好感排行", priority=5)
    async def cmd_rank(self, event: AstrMessageEvent, num: str = "10"):
        async for r in self.user_commands.show_favor_ranking(event, num): yield r

    @filter.command("负好感排行", priority=5)
    async def cmd_bad_rank(self, event: AstrMessageEvent, num: str = "10"):
        async for r in self.user_commands.show_negative_favor_ranking(event, num): yield r
        
    @filter.command("黑名单统计", priority=5)
    async def cmd_black_stats(self, event: AstrMessageEvent):
        async for r in self.user_commands.show_blacklist_stats(event): yield r

    @filter.command("缓存统计", priority=5)
    async def cmd_cache(self, event: AstrMessageEvent):
        async for r in self.user_commands.show_cache_stats(event): yield r

    @filter.command("设置情感", priority=5)
    async def cmd_set_emotion(self, event: AstrMessageEvent, user: str, dim: str, val: str):
        async for r in self.admin_commands.set_emotion(event, user, dim, val): yield r

    @filter.command("重置好感", priority=5)
    async def cmd_reset_favor(self, event: AstrMessageEvent, user: str):
        async for r in self.admin_commands.reset_favor(event, user): yield r

    @filter.command("查看好感", priority=5)
    async def cmd_view_favor(self, event: AstrMessageEvent, user: str):
        async for r in self.admin_commands.view_favor(event, user): yield r

    @filter.command("备份数据", priority=5)
    async def cmd_backup(self, event: AstrMessageEvent):
        async for r in self.admin_commands.backup_data(event): yield r

    async def terminate(self):
        if hasattr(self, 'auto_save_task'): self.auto_save_task.cancel()
        await self.user_manager.force_save()
        logger.info("EmotionAI 插件已安全关闭")