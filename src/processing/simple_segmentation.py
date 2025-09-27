#!/usr/bin/env python3
"""
简化的分句逻辑 - 替换复杂的main.py逻辑
======================================

核心原则：
1. spaCy基于语法分句（替代Whisper分句）
2. 强制长句检查（所有句子都要检查）
3. 强制时间戳分配（滑动窗口精确匹配）
4. Debug完整（匹配率统计）

流程：
Whisper数据提取 → spaCy分句 → 长句检查 → 时间戳分配
"""

import logging
import re
from typing import Dict, List, Tuple
import spacy

# 加载spaCy模型
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    raise Exception("spaCy model not found. Install: python -m spacy download en_core_web_sm")


def structure_and_split_segments(transcription_result: Dict, max_chars: int, segmentation_strategy: str = "rule_based") -> List[Dict]:
    """
    全新简化的分句架构
    
    Args:
        transcription_result: Whisper转录结果
        max_chars: 最大字符限制  
        segmentation_strategy: 分句策略（当前都用spaCy）
        
    Returns:
        List[Dict]: 处理好的字幕segments
    """
    logging.info("🚀 Starting simplified segmentation logic...")
    
    # 第一步：提取Whisper数据
    whisper_data = extract_whisper_data(transcription_result)
    if not whisper_data['all_words']:
        logging.error("❌ No word-level data from Whisper")
        return []
    
    # 第二步：spaCy语法分句  
    sentences = spacy_sentence_segmentation(whisper_data['all_words'])
    if not sentences:
        logging.error("❌ spaCy failed to generate sentences")
        return []
    
    logging.info(f"📝 spaCy generated {len(sentences)} initial sentences")
    
    # 第三步：强制长句检查和分割
    checked_sentences = mandatory_long_sentence_check(sentences, max_chars, whisper_data)
    
    # 第四步：强制时间戳分配和验证
    final_segments = mandatory_timestamp_assignment(checked_sentences, whisper_data)
    
    logging.info(f"✅ Final result: {len(final_segments)} segments")
    return final_segments


def extract_whisper_data(transcription_result: Dict) -> Dict:
    """提取Whisper的词级时间戳和原始segment边界"""
    whisper_segments = transcription_result.get("segments", [])
    if not whisper_segments:
        return {'all_words': [], 'segment_boundaries': []}
    
    all_words = []
    segment_boundaries = []
    
    for segment in whisper_segments:
        start_word_idx = len(all_words)
        segment_words = segment.get("words", [])
        
        if segment_words:  # 只处理有词的segment
            all_words.extend(segment_words)
            end_word_idx = len(all_words) - 1
            
            segment_boundaries.append({
                'start_word_idx': start_word_idx,
                'end_word_idx': end_word_idx,
                'original_text': segment.get('text', '').strip(),
                'start_time': segment.get('start', 0.0),
                'end_time': segment.get('end', 0.0)
            })
    
    logging.info(f"📊 Extracted {len(all_words)} words from {len(segment_boundaries)} Whisper segments")
    return {
        'all_words': all_words,
        'segment_boundaries': segment_boundaries
    }


def spacy_sentence_segmentation(all_words: List[Dict]) -> List[str]:
    """使用spaCy进行语法分句（忽略Whisper分句）"""
    
    # 合并所有Whisper words成完整文本
    word_texts = []
    for word_info in all_words:
        word = word_info.get('word', '').strip()
        if word:
            word_texts.append(word)
    
    if not word_texts:
        logging.error("No valid words found")
        return []
    
    full_text = " ".join(word_texts)
    logging.info(f"🔤 Combined text: {len(full_text)} chars")
    
    # spaCy语法分句
    try:
        doc = nlp(full_text)
        sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
        logging.info(f"📖 spaCy detected {len(sentences)} sentences")
        return sentences
    except Exception as e:
        logging.error(f"❌ spaCy processing failed: {e}")
        return []


def mandatory_long_sentence_check(sentences: List[str], max_chars: int, whisper_data: Dict) -> List[str]:
    """强制长句检查 - 所有句子都必须经过此步骤"""
    logging.info(f"\n🔍 MANDATORY LONG SENTENCE CHECK (max_chars: {max_chars})")
    logging.info(f"   Checking {len(sentences)} sentences...")
    
    checked_sentences = []
    split_count = 0
    
    for i, sentence in enumerate(sentences):
        sentence = sentence.strip()
        if not sentence:
            continue
            
        logging.info(f"   📏 Sentence {i+1}: {len(sentence)} chars")
        logging.info(f"      Text: '{sentence[:60]}{'...' if len(sentence) > 60 else ''}'")
        
        if len(sentence) <= max_chars:
            logging.info(f"      ✅ Length OK")
            checked_sentences.append(sentence)
        else:
            logging.warning(f"      ⚠️  TOO LONG! Splitting...")
            # 分割长句
            split_parts = split_long_sentence(sentence, max_chars, whisper_data)
            checked_sentences.extend(split_parts)
            split_count += 1
            logging.info(f"      ✂️  Split into {len(split_parts)} parts:")
            for j, part in enumerate(split_parts):
                logging.info(f"         {j+1}. ({len(part)} chars) '{part[:40]}{'...' if len(part) > 40 else ''}'")
    
    logging.info(f"📊 Long sentence check complete:")
    logging.info(f"   📥 Input:  {len(sentences)} sentences")
    logging.info(f"   📤 Output: {len(checked_sentences)} sentences")
    logging.info(f"   ✂️  Split:  {split_count} long sentences")
    
    return checked_sentences


def split_long_sentence(sentence: str, max_chars: int, whisper_data: Dict) -> List[str]:
    """分割长句 - 多种策略"""
    
    # 策略1: 在Whisper segment边界分割（优先）
    whisper_splits = try_split_at_whisper_boundaries(sentence, max_chars, whisper_data)
    if len(whisper_splits) > 1 and all(len(part) <= max_chars for part in whisper_splits):
        logging.info(f"         🎯 Using Whisper boundaries")
        return whisper_splits
    
    # 策略2: 语法点分割（连词、标点）
    grammar_splits = try_split_at_grammar_points(sentence, max_chars)
    if len(grammar_splits) > 1 and all(len(part) <= max_chars for part in grammar_splits):
        logging.info(f"         📝 Using grammar points")
        return grammar_splits
    
    # 策略3: 强制词边界分割（最后手段）
    word_splits = force_split_at_words(sentence, max_chars)
    logging.info(f"         ✂️  Force split at words")
    return word_splits


def try_split_at_whisper_boundaries(sentence: str, max_chars: int, whisper_data: Dict) -> List[str]:
    """尝试在Whisper原始segment边界分割"""
    # 简化版本：先返回原句，后续可以实现Whisper边界逻辑
    # TODO: 实现Whisper边界检测和分割
    return [sentence]


def try_split_at_grammar_points(sentence: str, max_chars: int) -> List[str]:
    """在语法点分割：连词、逗号、分号等"""
    
    # 定义分割点（按优先级）
    split_patterns = [
        r'\b(and|but|or|so|because|since|while|although|however|therefore|meanwhile)\b',
        r'[,;:]',
        r'\.',
    ]
    
    for pattern in split_patterns:
        matches = list(re.finditer(pattern, sentence, re.IGNORECASE))
        
        if matches:
            splits = []
            last_pos = 0
            
            for match in matches:
                split_pos = match.end()  # 在匹配后分割
                part = sentence[last_pos:split_pos].strip()
                
                if len(part) >= 20:  # 避免太短的片段
                    if len(part) <= max_chars:
                        splits.append(part)
                        last_pos = split_pos
                    else:
                        break  # 这个策略不适用
            
            # 添加剩余部分
            if last_pos < len(sentence):
                remaining = sentence[last_pos:].strip()
                if remaining:
                    splits.append(remaining)
            
            # 检查所有部分都符合长度要求
            if len(splits) > 1 and all(len(part) <= max_chars for part in splits):
                return splits
    
    return [sentence]


def force_split_at_words(sentence: str, max_chars: int) -> List[str]:
    """强制在词边界分割（保证结果）"""
    words = sentence.split()
    if not words:
        return [sentence]
    
    splits = []
    current_segment = []
    current_length = 0
    
    for word in words:
        word_length = len(word)
        space_length = 1 if current_segment else 0
        total_length = current_length + space_length + word_length
        
        if total_length > max_chars and current_segment:
            # 开始新segment
            splits.append(' '.join(current_segment))
            current_segment = [word]
            current_length = word_length
        else:
            current_segment.append(word)
            current_length = total_length
    
    # 添加最后一个segment
    if current_segment:
        splits.append(' '.join(current_segment))
    
    return splits


def mandatory_timestamp_assignment(sentences: List[str], whisper_data: Dict) -> List[Dict]:
    """强制时间戳分配 - 滑动窗口精确匹配 + Debug验证"""
    logging.info(f"\n⏱️  MANDATORY TIMESTAMP ASSIGNMENT")
    logging.info(f"   Assigning timestamps to {len(sentences)} sentences...")
    
    all_words = whisper_data['all_words']
    if not all_words:
        logging.error("❌ No word data for timestamp assignment")
        return []
    
    # 创建清理后的词序列用于匹配
    clean_word_sequence = []
    for i, word_info in enumerate(all_words):
        original_word = word_info.get('word', '').strip()
        if original_word:
            # 清理标点但保留缩写（I'm, don't等）
            clean_word = re.sub(r'[^\w\s\'-]', '', original_word).lower().strip()
            if clean_word:
                clean_word_sequence.append({
                    'clean_word': clean_word,
                    'original_word': original_word,
                    'start': word_info.get('start', 0.0),
                    'end': word_info.get('end', 0.0)
                })
        
        # 显示处理进度
        if (i + 1) % 100 == 0 or i == len(all_words) - 1:
            logging.info(f"   🔄 Processing words: {i + 1}/{len(all_words)}")
    
    logging.info(f"   📊 {len(clean_word_sequence)} words available for matching")
    
    final_segments = []
    used_word_indices = set()
    perfect_matches = 0
    good_matches = 0
    poor_matches = 0
    
    # 逐句匹配
    for i, sentence in enumerate(sentences):
        # logging.info(f"\n   🎯 Matching sentence {i+1}/{len(sentences)}")
        # logging.info(f"      Text: '{sentence[:50]}{'...' if len(sentence) > 50 else ''}'")
        
        # 简化进度显示
        if (i + 1) % 10 == 0 or i == len(sentences) - 1:
            logging.info(f"   🎯 Matching sentences: {i + 1}/{len(sentences)}")
        
        # 滑动窗口匹配
        start_idx, end_idx, confidence = sliding_window_match(sentence, clean_word_sequence, used_word_indices)
        
        if start_idx != -1 and end_idx != -1:
            # 标记已使用的词
            for idx in range(start_idx, end_idx + 1):
                used_word_indices.add(idx)
            
            # 创建segment
            segment = {
                'text': sentence,
                'start': clean_word_sequence[start_idx]['start'],
                'end': clean_word_sequence[end_idx]['end'],
                'word_count': end_idx - start_idx + 1,
                'timing_confidence': confidence
            }
            final_segments.append(segment)
            
            # 统计匹配质量
            if confidence == 1.0:
                perfect_matches += 1
                # emoji = "✅"
            elif confidence >= 0.8:
                good_matches += 1
                # emoji = "🟡"
            else:
                poor_matches += 1
                # emoji = "🟠"
            
            # 注释掉详细匹配信息
            # logging.info(f"      {emoji} Match: words [{start_idx}-{end_idx}], confidence: {confidence:.2f}")
            # logging.info(f"         Time: {segment['start']:.2f}s - {segment['end']:.2f}s")
        else:
            # 匹配失败，创建fallback segment并输出详细debug信息
            logging.error(f"\n❌ MATCH FAILED for sentence {i+1}/{len(sentences)}:")
            logging.error(f"   原句: '{sentence}'")
            
            # 提取句子的清理词用于debug
            sentence_clean_words = []
            for word in sentence.split():
                clean_word = re.sub(r'[^\w\s\'-]', '', word).lower().strip()
                if clean_word:
                    sentence_clean_words.append(clean_word)
            
            logging.error(f"   句子清理后的词: {sentence_clean_words}")
            
            # 显示附近可用的词序列用于分析
            available_start = max(0, len([idx for idx in range(len(clean_word_sequence)) if idx not in used_word_indices]) - 10)
            available_words = [clean_word_sequence[i]['clean_word'] for i in range(len(clean_word_sequence)) if i not in used_word_indices]
            if available_words:
                logging.error(f"   可用词序列(前20个): {available_words[:20]}")
            else:
                logging.error(f"   ⚠️  所有词都已被使用!")
            
            fallback_segment = {
                'text': sentence,
                'start': 0.0,
                'end': 0.0,
                'word_count': len(sentence.split()),
                'timing_confidence': 0.0
            }
            final_segments.append(fallback_segment)
            poor_matches += 1
    
    # 统计和报告
    total_sentences = len(sentences)
    success_rate = (perfect_matches + good_matches) / total_sentences if total_sentences > 0 else 0
    
    logging.info(f"\n📊 TIMESTAMP MATCHING RESULTS:")
    logging.info(f"   🎯 Perfect matches (1.0):   {perfect_matches}/{total_sentences}")
    logging.info(f"   🟡 Good matches (≥0.8):     {good_matches}/{total_sentences}")
    logging.info(f"   🟠 Poor matches (<0.8):     {poor_matches}/{total_sentences}")
    logging.info(f"   ❌ Failed matches (0.0):    {total_sentences - perfect_matches - good_matches - poor_matches}/{total_sentences}")
    logging.info(f"   📈 Overall success rate:    {success_rate:.1%}")
    
    if success_rate == 1.0:
        logging.info(f"   🎉 PERFECT! All sentences matched with high confidence!")
    elif success_rate >= 0.9:
        logging.info(f"   ✅ Excellent matching quality")
    elif success_rate >= 0.8:
        logging.info(f"   🟡 Good matching quality")
    else:
        logging.warning(f"   ⚠️  Poor matching quality - may need debugging")
    
    return final_segments


def sliding_window_match(sentence: str, word_sequence: List[Dict], used_indices: set) -> Tuple[int, int, float]:
    """滑动窗口匹配算法 - 精确匹配句子到词序列"""
    
    # 提取句子中的清理词
    sentence_words = []
    for word in sentence.split():
        clean_word = re.sub(r'[^\w\s\'-]', '', word).lower().strip()
        if clean_word:
            sentence_words.append(clean_word)
    
    if not sentence_words:
        return -1, -1, 0.0
    
    # logging.info(f"         🔍 Searching for {len(sentence_words)} words: {sentence_words[:5]}{'...' if len(sentence_words) > 5 else ''}")
    
    best_start = -1
    best_end = -1
    best_confidence = 0.0
    
    # 滑动窗口搜索
    search_end = len(word_sequence) - len(sentence_words) + 1
    for start_idx in range(search_end):
        end_idx = start_idx + len(sentence_words) - 1
        
        # 跳过已使用的区域
        if any(idx in used_indices for idx in range(start_idx, end_idx + 1)):
            continue
        
        # 计算匹配度
        matches = 0
        mismatches = []
        
        for i, sentence_word in enumerate(sentence_words):
            word_idx = start_idx + i
            if word_idx < len(word_sequence):
                if word_sequence[word_idx]['clean_word'] == sentence_word:
                    matches += 1
                else:
                    mismatches.append(f"{sentence_word}≠{word_sequence[word_idx]['clean_word']}")
        
        confidence = matches / len(sentence_words)
        
        # 完美匹配，立即返回
        if confidence == 1.0:
            # logging.info(f"         🎯 Perfect match found at words [{start_idx}-{end_idx}]")
            return start_idx, end_idx, confidence
        
        # 记录最佳部分匹配
        if confidence > best_confidence:
            best_confidence = confidence
            best_start = start_idx
            best_end = end_idx
    
    # 只接受高置信度的匹配
    if best_confidence >= 0.8:
        # logging.info(f"         🟡 Good partial match: {best_confidence:.2f} at words [{best_start}-{best_end}]")
        return best_start, best_end, best_confidence
    elif best_confidence > 0.0:
        # logging.warning(f"         🟠 Poor match: {best_confidence:.2f} (rejected)")
        return -1, -1, best_confidence
    else:
        # logging.error(f"         ❌ No match found")
        return -1, -1, 0.0


if __name__ == "__main__":
    print("简化分句逻辑已准备好！")
    print("特点：")
    print("✅ spaCy语法分句（高质量）")
    print("✅ 强制长句检查（所有句子）")
    print("✅ 强制时间戳分配（精确匹配）")
    print("✅ 完整Debug信息（匹配率统计）")
    print("✅ 架构简洁（无补丁代码）")