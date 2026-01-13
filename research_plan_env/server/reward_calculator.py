"""Reward calculator for evaluating research plans against rubric criteria.

Uses FLAN-T5 for LLM-based evaluation of how well a research plan addresses
each rubric criterion, combined with semantic similarity and format metrics.
"""

import re
from enum import Enum
from typing import Optional

import nltk
import torch
from nltk.corpus import words as nltk_words
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# Ensure NLTK words corpus is downloaded
try:
    nltk.data.find("corpora/words")
except LookupError:
    nltk.download("words", quiet=True)


class RatingChoice(str, Enum):
    """Structured rating choices for criterion evaluation."""

    EXCELLENT = "excellent"
    GOOD = "good"
    PARTIAL = "partial"
    POOR = "poor"


class RewardCalculator:
    """Calculates reward for research plan submissions.

    Uses FLAN-T5-small with outlines for structured LLM-based rubric evaluation,
    combined with rule-based length and format scoring.
    """

    # Target length range for research plans (in words)
    MIN_WORDS = 200
    OPTIMAL_MIN_WORDS = 400
    OPTIMAL_MAX_WORDS = 1500
    MAX_WORDS = 3000

    # Reward weights
    RUBRIC_WEIGHT = 0.6
    LENGTH_WEIGHT = 0.2
    FORMAT_WEIGHT = 0.2

    # Score mapping for ratings
    SCORE_MAP = {
        RatingChoice.EXCELLENT: 1.0,
        RatingChoice.GOOD: 0.75,
        RatingChoice.PARTIAL: 0.5,
        RatingChoice.POOR: 0.25,
    }

    # English dictionary for coherence checking
    ENGLISH_WORDS = frozenset(w.lower() for w in nltk_words.words())

    def __init__(
        self,
        model_name: str = "google/flan-t5-small",
        embedding_model_name: str = "all-MiniLM-L6-v2",
        device: Optional[str] = None,
    ):
        """Initialise the reward calculator.

        Args:
            model_name: HuggingFace model identifier for the evaluation LLM.
            embedding_model_name: Sentence transformer model for semantic similarity.
            device: Device to run the model on (auto-detected if None).

        Raises:
            RuntimeError: If the LLM or embedding model cannot be loaded.
        """
        self.model_name = model_name

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.hf_model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
            self.hf_model.to(self.device)
            self.hf_model.eval()
        except Exception as e:
            raise RuntimeError(f"Failed to load LLM '{model_name}': {e}") from e

        # Load sentence transformer for semantic similarity
        try:
            self.embedder = SentenceTransformer(embedding_model_name, device=self.device)
        except Exception as e:
            raise RuntimeError(
                f"Failed to load embedding model '{embedding_model_name}': {e}"
            ) from e

        # Valid rating choices for parsing LLM output
        self._valid_ratings = {"excellent", "good", "partial", "poor"}

    def _check_coherence(self, text: str) -> float:
        """Check if text appears to be coherent and meaningful.

        Uses a combination of:
        1. Dictionary validation - what proportion of words are real English
        2. Semantic coherence - do sentences relate to each other

        Returns a score from 0.0 (nonsense) to 1.0 (coherent).
        """
        if not text or not text.strip():
            return 0.0

        # Extract alphabetic words
        tokens = re.findall(r"\b[a-zA-Z]+\b", text.lower())
        if len(tokens) < 10:
            return 0.2

        # Check for excessive repetition (sign of nonsense)
        unique_tokens = set(tokens)
        if len(unique_tokens) / len(tokens) < 0.3:
            return 0.1

        # Dictionary check: what proportion of words are real English
        valid_words = sum(1 for w in tokens if w in self.ENGLISH_WORDS)
        dictionary_score = valid_words / len(tokens)

        # If mostly gibberish, return low score immediately
        if dictionary_score < 0.5:
            return dictionary_score * 0.4

        # Semantic coherence: check if sentences are related
        sentences = re.split(r"[.!?]+", text)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 20]

        if len(sentences) >= 3:
            # Embed sentences and check variance
            embeddings = self.embedder.encode(sentences[:10])  # Limit for efficiency
            # Calculate pairwise similarities
            similarities = cosine_similarity(embeddings)
            # Average off-diagonal similarity (how related sentences are)
            n = len(similarities)
            if n > 1:
                total_sim = sum(
                    similarities[i][j] for i in range(n) for j in range(n) if i != j
                )
                avg_similarity = total_sim / (n * (n - 1))
                semantic_score = float(max(0.0, min(1.0, avg_similarity)))
            else:
                semantic_score = 0.7  # Single sentence, assume OK
        else:
            semantic_score = 0.7  # Too few sentences to check

        # Combine scores: dictionary matters more for catching gibberish
        return 0.6 * dictionary_score + 0.4 * semantic_score

    def _calculate_relevance(self, plan: str, criterion: str) -> float:
        """Calculate semantic relevance between the plan and criterion.

        Uses sentence embeddings to capture semantic similarity rather than
        simple keyword matching. This handles synonyms, paraphrasing, and
        conceptually related content.

        Returns a score from 0.0 to 1.0.
        """
        if not plan.strip() or not criterion.strip():
            return 0.0

        # For long plans, use a representative chunk to avoid embedding truncation
        max_plan_chars = 2000
        plan_text = plan[:max_plan_chars] if len(plan) > max_plan_chars else plan

        embeddings = self.embedder.encode([plan_text, criterion])
        similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]

        # Cosine similarity ranges from -1 to 1, but for text it's typically 0-1
        # Clamp and return
        return float(max(0.0, min(1.0, similarity)))

    def calculate_reward(
        self,
        plan: str,
        rubric: list[str],
        goal: str,
        reference_solution: Optional[str] = None,
    ) -> tuple[float, dict]:
        """Calculate the total reward for a research plan.

        Args:
            plan: The submitted research plan text.
            rubric: List of rubric criteria to evaluate against.
            goal: The research goal/task description.
            reference_solution: Optional reference solution for comparison.

        Returns:
            Tuple of (total_reward, breakdown_dict) where breakdown contains:
            - rubric_scores: dict mapping rubric index to score (0.0-1.0)
            - length_score: float (0.0-1.0)
            - format_score: float (0.0-1.0)
            - rubric_avg: float (0.0-1.0) average rubric score
            - feedback: str with detailed evaluation feedback
        """
        # Check coherence first
        coherence = self._check_coherence(plan)
        if coherence < 0.3:
            # Nonsense input - give minimal scores
            rubric_scores = {i: 0.1 for i in range(len(rubric))}
            return 0.1, {
                "rubric_scores": rubric_scores,
                "rubric_avg": 0.1,
                "length_score": 0.0,
                "format_score": 0.0,
                "feedback": "The submission does not appear to be a coherent research plan. "
                "Please provide a meaningful response that addresses the research goal.",
            }

        # Evaluate each rubric criterion
        rubric_scores = {}
        rubric_feedback = []

        for i, criterion in enumerate(rubric):
            score = self._evaluate_criterion(plan, criterion, goal)
            rubric_scores[i] = score
            rubric_feedback.append(
                f"Criterion {i + 1}: {score:.2f}/1.0 - {criterion[:100]}..."
            )

        rubric_avg = sum(rubric_scores.values()) / len(rubric_scores) if rubric_scores else 0.0

        # Apply coherence penalty
        rubric_avg *= coherence

        # Calculate length score
        length_score = self._calculate_length_score(plan)

        # Calculate format score
        format_score = self._calculate_format_score(plan)

        # Compute total weighted reward
        total_reward = (
            self.RUBRIC_WEIGHT * rubric_avg
            + self.LENGTH_WEIGHT * length_score
            + self.FORMAT_WEIGHT * format_score
        )

        # Generate feedback
        feedback = self._generate_feedback(
            rubric_avg, length_score, format_score, rubric_feedback, plan
        )

        breakdown = {
            "rubric_scores": rubric_scores,
            "rubric_avg": rubric_avg,
            "length_score": length_score,
            "format_score": format_score,
            "feedback": feedback,
        }

        return total_reward, breakdown

    def _evaluate_criterion(self, plan: str, criterion: str, goal: str) -> float:
        """Evaluate how well the plan addresses a specific criterion.

        Uses a combination of keyword relevance and LLM evaluation.

        Args:
            plan: The research plan text.
            criterion: The rubric criterion to evaluate.
            goal: The research goal for context.

        Returns:
            Score from 0.0 to 1.0.

        Raises:
            RuntimeError: If the LLM fails to generate a valid rating.
        """
        # First check keyword relevance
        relevance = self._calculate_relevance(plan, criterion)

        # If very low relevance, don't bother with LLM - return low score
        if relevance < 0.1:
            return 0.1

        # Get LLM evaluation
        llm_score = self._llm_evaluate_criterion(plan, criterion, goal)

        # Combine relevance and LLM score
        # If relevance is low, cap the maximum score
        if relevance < 0.3:
            return min(llm_score, 0.5) * relevance * 2
        else:
            # Weight LLM score by relevance
            return llm_score * (0.5 + 0.5 * relevance)

    def _llm_evaluate_criterion(self, plan: str, criterion: str, goal: str) -> float:
        """Use LLM to evaluate criterion satisfaction.

        The model is prompted to rate how well the plan addresses the criterion
        on a scale, with the response parsed to extract a score.
        """
        # Truncate plan to fit in context
        max_plan_chars = 1500
        truncated_plan = plan[:max_plan_chars] + ("..." if len(plan) > max_plan_chars else "")

        prompt = f"""You are evaluating a research plan. Be critical and objective.

Research Goal: {goal[:400]}

Criterion to evaluate: {criterion}

Research Plan:
{truncated_plan}

Rating guide:
- excellent: Plan explicitly and thoroughly addresses this specific criterion with concrete details
- good: Plan addresses this criterion but lacks some detail or specificity
- partial: Plan only tangentially mentions concepts related to this criterion
- poor: Plan does not address this criterion or is irrelevant

Does this plan address the criterion? Rate as excellent, good, partial, or poor:"""

        try:
            # Generate rating using T5
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                max_length=512,
                truncation=True,
            ).to(self.device)

            with torch.no_grad():
                outputs = self.hf_model.generate(
                    **inputs,
                    max_new_tokens=10,
                    do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id,
                )

            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True).lower().strip()

            # Parse the rating from the response
            for rating in self._valid_ratings:
                if rating in response:
                    return self.SCORE_MAP[RatingChoice(rating)]

            # Default to partial if no valid rating found
            return self.SCORE_MAP[RatingChoice.PARTIAL]

        except Exception as e:
            raise RuntimeError(
                f"Failed to evaluate criterion '{criterion[:50]}...': {e}"
            ) from e

    def _calculate_length_score(self, plan: str) -> float:
        """Calculate score based on plan length appropriateness.

        Research plans should be substantial but not excessively long.
        Optimal range is 400-1500 words.
        """
        word_count = len(plan.split())

        if word_count < self.MIN_WORDS:
            # Too short - linear penalty
            return max(0.0, word_count / self.MIN_WORDS * 0.5)
        elif word_count < self.OPTIMAL_MIN_WORDS:
            # Below optimal but acceptable
            progress = (word_count - self.MIN_WORDS) / (self.OPTIMAL_MIN_WORDS - self.MIN_WORDS)
            return 0.5 + 0.5 * progress
        elif word_count <= self.OPTIMAL_MAX_WORDS:
            # Optimal range
            return 1.0
        elif word_count <= self.MAX_WORDS:
            # Above optimal but acceptable
            progress = (word_count - self.OPTIMAL_MAX_WORDS) / (self.MAX_WORDS - self.OPTIMAL_MAX_WORDS)
            return 1.0 - 0.3 * progress
        else:
            # Too long
            excess = (word_count - self.MAX_WORDS) / self.MAX_WORDS
            return max(0.3, 0.7 - excess * 0.4)

    def _calculate_format_score(self, plan: str) -> float:
        """Calculate score based on plan structure.

        Good research plans should have clear organisation:
        - Multiple paragraphs (separated by blank lines)
        - Section headers or structural markers
        - Lists for enumerating points
        """
        score = 0.0

        # Check for paragraphs (separated by blank lines)
        paragraphs = [p.strip() for p in plan.split("\n\n") if p.strip()]
        if len(paragraphs) >= 5:
            score += 0.4
        elif len(paragraphs) >= 3:
            score += 0.25
        elif len(paragraphs) >= 2:
            score += 0.15

        # Check for section headers (markdown style or numbered sections)
        lines = plan.split("\n")
        header_patterns = [
            r"^#{1,3}\s+\w",  # Markdown headers
            r"^\d+\.\s+[A-Z]",  # Numbered sections like "1. Introduction"
            r"^[A-Z][A-Za-z\s]{2,30}:$",  # Title case headers ending with colon
        ]
        header_count = sum(
            1 for line in lines
            if any(re.match(pat, line.strip()) for pat in header_patterns)
        )
        if header_count >= 4:
            score += 0.35
        elif header_count >= 2:
            score += 0.2
        elif header_count >= 1:
            score += 0.1

        # Check for lists (bullet points or numbered items)
        list_patterns = [
            r"^\s*[-*+]\s+\w",  # Bullet points
            r"^\s*\d+[.)]\s+\w",  # Numbered lists
            r"^\s*[a-z][.)]\s+\w",  # Lettered lists
        ]
        list_items = sum(
            1 for line in lines
            if any(re.match(pat, line) for pat in list_patterns)
        )
        if list_items >= 5:
            score += 0.25
        elif list_items >= 3:
            score += 0.15
        elif list_items >= 1:
            score += 0.05

        return min(1.0, score)

    def _generate_feedback(
        self,
        rubric_avg: float,
        length_score: float,
        format_score: float,
        rubric_feedback: list[str],
        plan: str,
    ) -> str:
        """Generate human-readable feedback for the evaluation."""
        word_count = len(plan.split())

        feedback_parts = []

        # Overall assessment
        total = (
            self.RUBRIC_WEIGHT * rubric_avg
            + self.LENGTH_WEIGHT * length_score
            + self.FORMAT_WEIGHT * format_score
        )
        if total >= 0.8:
            feedback_parts.append("Excellent research plan overall.")
        elif total >= 0.6:
            feedback_parts.append("Good research plan with room for improvement.")
        elif total >= 0.4:
            feedback_parts.append("Adequate plan but several areas need strengthening.")
        else:
            feedback_parts.append("Plan needs significant improvement.")

        # Rubric feedback
        feedback_parts.append(f"\nRubric Coverage ({rubric_avg:.0%}):")
        for rf in rubric_feedback:
            feedback_parts.append(f"  - {rf}")

        # Length feedback
        feedback_parts.append(f"\nLength ({length_score:.0%}): {word_count} words")
        if word_count < self.OPTIMAL_MIN_WORDS:
            feedback_parts.append(f"  Consider expanding to at least {self.OPTIMAL_MIN_WORDS} words.")
        elif word_count > self.OPTIMAL_MAX_WORDS:
            feedback_parts.append(f"  Consider condensing to under {self.OPTIMAL_MAX_WORDS} words.")

        # Format feedback
        feedback_parts.append(f"\nStructure ({format_score:.0%}):")
        if format_score < 0.5:
            feedback_parts.append("  Add clear sections, headers, and bullet points for better organisation.")

        return "\n".join(feedback_parts)
