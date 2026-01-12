from http.server import BaseHTTPRequestHandler
import json
import requests
import numpy as np
from collections import Counter, deque
from scipy import stats
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
import warnings
import hashlib
import time
from datetime import datetime
warnings.filterwarnings('ignore')

# ============================================
# 🔥 100+ AI/ML ALGORITHMS ENSEMBLE PREDICTOR
# ============================================

class UltraPredictionEngine:
    def __init__(self):
        self.prediction_cache = {}
        self.history = []
        self.win_count = 0
        self.loss_count = 0
        self.last_prediction = None
        self.last_period = None
        
    def fetch_data(self):
        """Fetch latest data from API"""
        try:
            url = "https://draw.ar-lottery01.com/WinGo/WinGo_1M/GetHistoryIssuePage.json"
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
                'Accept': 'application/json',
                'Cache-Control': 'no-cache'
            }
            response = requests.get(url, headers=headers, timeout=10)
            data = response.json()
            return data
        except Exception as e:
            return {"error": str(e)}
    
    def extract_results(self, data):
        """Extract numbers and results from API data"""
        results = []
        try:
            if isinstance(data, dict) and 'data' in data:
                items = data['data'].get('list', []) or data['data'].get('items', [])
            elif isinstance(data, list):
                items = data
            else:
                items = data.get('list', []) or data.get('items', [])
            
            for item in items[:100]:  # Last 100 results
                if isinstance(item, dict):
                    number = item.get('number', item.get('num', item.get('result', 0)))
                    period = item.get('issueNumber', item.get('period', item.get('issue', '')))
                    if number is not None:
                        num = int(str(number)[-1]) if number else 0
                        result = 'SMALL' if num < 5 else 'BIG'
                        results.append({
                            'period': period,
                            'number': num,
                            'result': result
                        })
        except:
            pass
        return results

    # ============================================
    # 📊 ALGORITHM 1-10: MOVING AVERAGES
    # ============================================
    
    def algo_sma_3(self, numbers):
        """Simple Moving Average 3"""
        if len(numbers) < 3:
            return 0.5
        return np.mean(numbers[-3:]) / 9
    
    def algo_sma_5(self, numbers):
        """Simple Moving Average 5"""
        if len(numbers) < 5:
            return 0.5
        return np.mean(numbers[-5:]) / 9
    
    def algo_sma_7(self, numbers):
        """Simple Moving Average 7"""
        if len(numbers) < 7:
            return 0.5
        return np.mean(numbers[-7:]) / 9
    
    def algo_sma_10(self, numbers):
        """Simple Moving Average 10"""
        if len(numbers) < 10:
            return 0.5
        return np.mean(numbers[-10:]) / 9
    
    def algo_sma_15(self, numbers):
        """Simple Moving Average 15"""
        if len(numbers) < 15:
            return 0.5
        return np.mean(numbers[-15:]) / 9
    
    def algo_ema_3(self, numbers):
        """Exponential Moving Average 3"""
        if len(numbers) < 3:
            return 0.5
        weights = np.exp(np.linspace(-1, 0, 3))
        return np.average(numbers[-3:], weights=weights) / 9
    
    def algo_ema_5(self, numbers):
        """Exponential Moving Average 5"""
        if len(numbers) < 5:
            return 0.5
        weights = np.exp(np.linspace(-1, 0, 5))
        return np.average(numbers[-5:], weights=weights) / 9
    
    def algo_ema_7(self, numbers):
        """Exponential Moving Average 7"""
        if len(numbers) < 7:
            return 0.5
        weights = np.exp(np.linspace(-1, 0, 7))
        return np.average(numbers[-7:], weights=weights) / 9
    
    def algo_wma_5(self, numbers):
        """Weighted Moving Average 5"""
        if len(numbers) < 5:
            return 0.5
        weights = [1, 2, 3, 4, 5]
        return np.average(numbers[-5:], weights=weights) / 9
    
    def algo_wma_10(self, numbers):
        """Weighted Moving Average 10"""
        if len(numbers) < 10:
            return 0.5
        weights = list(range(1, 11))
        return np.average(numbers[-10:], weights=weights) / 9

    # ============================================
    # 📊 ALGORITHM 11-20: STATISTICAL ANALYSIS
    # ============================================
    
    def algo_mean_reversion(self, numbers):
        """Mean Reversion Strategy"""
        if len(numbers) < 10:
            return 0.5
        mean = np.mean(numbers)
        last = numbers[-1]
        return 0.7 if last < mean else 0.3
    
    def algo_std_deviation(self, numbers):
        """Standard Deviation Analysis"""
        if len(numbers) < 10:
            return 0.5
        std = np.std(numbers)
        mean = np.mean(numbers)
        last = numbers[-1]
        z_score = (last - mean) / (std + 0.001)
        return 0.5 + (z_score * 0.1)
    
    def algo_variance_analysis(self, numbers):
        """Variance Analysis"""
        if len(numbers) < 10:
            return 0.5
        var = np.var(numbers[-10:])
        return 0.5 + (var / 100)
    
    def algo_skewness(self, numbers):
        """Skewness Analysis"""
        if len(numbers) < 10:
            return 0.5
        skew = stats.skew(numbers[-10:])
        return 0.5 + (skew * 0.1)
    
    def algo_kurtosis(self, numbers):
        """Kurtosis Analysis"""
        if len(numbers) < 10:
            return 0.5
        kurt = stats.kurtosis(numbers[-10:])
        return 0.5 + (kurt * 0.05)
    
    def algo_median_analysis(self, numbers):
        """Median Analysis"""
        if len(numbers) < 5:
            return 0.5
        median = np.median(numbers[-10:])
        return median / 9
    
    def algo_mode_analysis(self, numbers):
        """Mode Analysis"""
        if len(numbers) < 5:
            return 0.5
        mode_result = stats.mode(numbers[-10:], keepdims=True)
        return mode_result.mode[0] / 9
    
    def algo_percentile_25(self, numbers):
        """25th Percentile Analysis"""
        if len(numbers) < 10:
            return 0.5
        p25 = np.percentile(numbers[-20:], 25)
        return p25 / 9
    
    def algo_percentile_75(self, numbers):
        """75th Percentile Analysis"""
        if len(numbers) < 10:
            return 0.5
        p75 = np.percentile(numbers[-20:], 75)
        return p75 / 9
    
    def algo_iqr_analysis(self, numbers):
        """Interquartile Range Analysis"""
        if len(numbers) < 10:
            return 0.5
        q1, q3 = np.percentile(numbers[-20:], [25, 75])
        iqr = q3 - q1
        return 0.5 + (iqr / 20)

    # ============================================
    # 📊 ALGORITHM 21-30: PATTERN RECOGNITION
    # ============================================
    
    def algo_streak_analysis(self, results):
        """Streak Analysis"""
        if len(results) < 3:
            return 0.5
        streak = 1
        last_result = results[-1]
        for i in range(len(results)-2, -1, -1):
            if results[i] == last_result:
                streak += 1
            else:
                break
        # After long streak, predict opposite
        return 0.3 if last_result == 'BIG' and streak >= 3 else 0.7 if streak >= 3 else 0.5
    
    def algo_alternating_pattern(self, results):
        """Alternating Pattern Detection"""
        if len(results) < 4:
            return 0.5
        alternating = True
        for i in range(len(results)-3, len(results)-1):
            if results[i] == results[i+1]:
                alternating = False
                break
        if alternating:
            return 0.3 if results[-1] == 'BIG' else 0.7
        return 0.5
    
    def algo_triple_pattern(self, results):
        """Triple Pattern Detection"""
        if len(results) < 6:
            return 0.5
        pattern = results[-3:]
        prev_pattern = results[-6:-3]
        if pattern == prev_pattern:
            return 0.3 if results[-1] == 'BIG' else 0.7
        return 0.5
    
    def algo_double_pattern(self, results):
        """Double Pattern Detection"""
        if len(results) < 4:
            return 0.5
        if results[-1] == results[-2]:
            return 0.3 if results[-1] == 'BIG' else 0.7
        return 0.5
    
    def algo_pattern_5(self, results):
        """5-Length Pattern Analysis"""
        if len(results) < 10:
            return 0.5
        pattern = tuple(results[-5:])
        for i in range(len(results)-10, 0, -1):
            if tuple(results[i:i+5]) == pattern:
                if i + 5 < len(results):
                    return 0.7 if results[i+5] == 'BIG' else 0.3
        return 0.5
    
    def algo_fibonacci_pattern(self, numbers):
        """Fibonacci Pattern Analysis"""
        if len(numbers) < 5:
            return 0.5
        fib = [1, 1, 2, 3, 5, 8]
        last_5 = numbers[-5:]
        fib_match = sum(1 for n in last_5 if n in fib)
        return 0.5 + (fib_match * 0.05)
    
    def algo_prime_pattern(self, numbers):
        """Prime Number Pattern"""
        if len(numbers) < 5:
            return 0.5
        primes = [2, 3, 5, 7]
        prime_count = sum(1 for n in numbers[-5:] if n in primes)
        return 0.5 + (prime_count * 0.05)
    
    def algo_even_odd_pattern(self, numbers):
        """Even-Odd Pattern Analysis"""
        if len(numbers) < 5:
            return 0.5
        even_count = sum(1 for n in numbers[-5:] if n % 2 == 0)
        return even_count / 5
    
    def algo_high_low_pattern(self, numbers):
        """High-Low Pattern Analysis"""
        if len(numbers) < 5:
            return 0.5
        high_count = sum(1 for n in numbers[-5:] if n >= 5)
        return high_count / 5
    
    def algo_gap_analysis(self, numbers):
        """Gap Analysis Between Numbers"""
        if len(numbers) < 5:
            return 0.5
        gaps = [abs(numbers[i] - numbers[i-1]) for i in range(1, len(numbers[-5:]))]
        avg_gap = np.mean(gaps)
        return 0.5 + (avg_gap / 20)

    # ============================================
    # 📊 ALGORITHM 31-40: FREQUENCY ANALYSIS
    # ============================================
    
    def algo_frequency_small(self, results):
        """Frequency of SMALL in last 20"""
        if len(results) < 10:
            return 0.5
        small_count = sum(1 for r in results[-20:] if r == 'SMALL')
        return small_count / 20
    
    def algo_frequency_big(self, results):
        """Frequency of BIG in last 20"""
        if len(results) < 10:
            return 0.5
        big_count = sum(1 for r in results[-20:] if r == 'BIG')
        return big_count / 20
    
    def algo_hot_numbers(self, numbers):
        """Hot Numbers Analysis"""
        if len(numbers) < 20:
            return 0.5
        counter = Counter(numbers[-20:])
        hot = counter.most_common(3)
        hot_sum = sum(n for n, _ in hot)
        return hot_sum / 27
    
    def algo_cold_numbers(self, numbers):
        """Cold Numbers Analysis"""
        if len(numbers) < 20:
            return 0.5
        counter = Counter(numbers[-20:])
        cold = counter.most_common()[-3:]
        cold_sum = sum(n for n, _ in cold)
        return cold_sum / 27
    
    def algo_due_numbers(self, numbers):
        """Due Numbers Analysis"""
        if len(numbers) < 30:
            return 0.5
        all_nums = set(range(10))
        recent = set(numbers[-10:])
        due = all_nums - recent
        if due:
            avg_due = np.mean(list(due))
            return avg_due / 9
        return 0.5
    
    def algo_repeat_frequency(self, numbers):
        """Repeat Frequency Analysis"""
        if len(numbers) < 10:
            return 0.5
        repeats = sum(1 for i in range(1, len(numbers[-10:])) if numbers[-10:][i] == numbers[-10:][i-1])
        return 0.5 + (repeats * 0.05)
    
    def algo_cluster_analysis(self, numbers):
        """Cluster Analysis"""
        if len(numbers) < 10:
            return 0.5
        clusters = {'low': 0, 'mid': 0, 'high': 0}
        for n in numbers[-10:]:
            if n < 3:
                clusters['low'] += 1
            elif n < 7:
                clusters['mid'] += 1
            else:
                clusters['high'] += 1
        return clusters['high'] / 10
    
    def algo_distribution_chi2(self, numbers):
        """Chi-Square Distribution Test"""
        if len(numbers) < 20:
            return 0.5
        observed = [numbers[-20:].count(i) for i in range(10)]
        expected = [2] * 10
        chi2, _ = stats.chisquare(observed, expected)
        return 0.5 + (chi2 / 100)
    
    def algo_entropy_analysis(self, numbers):
        """Entropy Analysis"""
        if len(numbers) < 10:
            return 0.5
        counter = Counter(numbers[-10:])
        probs = [c/10 for c in counter.values()]
        entropy = -sum(p * np.log2(p + 0.001) for p in probs)
        return entropy / 4
    
    def algo_gini_coefficient(self, numbers):
        """Gini Coefficient Analysis"""
        if len(numbers) < 10:
            return 0.5
        sorted_nums = sorted(numbers[-10:])
        n = len(sorted_nums)
        cumsum = np.cumsum(sorted_nums)
        gini = (n + 1 - 2 * sum(cumsum) / cumsum[-1]) / n
        return 0.5 + gini

    # ============================================
    # 📊 ALGORITHM 41-50: MOMENTUM INDICATORS
    # ============================================
    
    def algo_rsi(self, numbers):
        """Relative Strength Index"""
        if len(numbers) < 14:
            return 0.5
        changes = [numbers[i] - numbers[i-1] for i in range(1, len(numbers[-14:]))]
        gains = [c for c in changes if c > 0]
        losses = [-c for c in changes if c < 0]
        avg_gain = np.mean(gains) if gains else 0.001
        avg_loss = np.mean(losses) if losses else 0.001
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi / 100
    
    def algo_macd(self, numbers):
        """MACD Indicator"""
        if len(numbers) < 12:
            return 0.5
        ema12 = np.mean(numbers[-12:])
        ema26 = np.mean(numbers[-min(26, len(numbers)):])
        macd = ema12 - ema26
        return 0.5 + (macd / 10)
    
    def algo_stochastic(self, numbers):
        """Stochastic Oscillator"""
        if len(numbers) < 14:
            return 0.5
        low = min(numbers[-14:])
        high = max(numbers[-14:])
        current = numbers[-1]
        if high - low == 0:
            return 0.5
        k = (current - low) / (high - low)
        return k
    
    def algo_momentum_3(self, numbers):
        """3-Period Momentum"""
        if len(numbers) < 4:
            return 0.5
        momentum = numbers[-1] - numbers[-4]
        return 0.5 + (momentum / 20)
    
    def algo_momentum_5(self, numbers):
        """5-Period Momentum"""
        if len(numbers) < 6:
            return 0.5
        momentum = numbers[-1] - numbers[-6]
        return 0.5 + (momentum / 20)
    
    def algo_momentum_10(self, numbers):
        """10-Period Momentum"""
        if len(numbers) < 11:
            return 0.5
        momentum = numbers[-1] - numbers[-11]
        return 0.5 + (momentum / 20)
    
    def algo_roc(self, numbers):
        """Rate of Change"""
        if len(numbers) < 10:
            return 0.5
        roc = ((numbers[-1] - numbers[-10]) / (numbers[-10] + 0.001)) * 100
        return 0.5 + (roc / 200)
    
    def algo_williams_r(self, numbers):
        """Williams %R"""
        if len(numbers) < 14:
            return 0.5
        high = max(numbers[-14:])
        low = min(numbers[-14:])
        if high - low == 0:
            return 0.5
        wr = (high - numbers[-1]) / (high - low) * -100
        return (wr + 100) / 100
    
    def algo_cci(self, numbers):
        """Commodity Channel Index"""
        if len(numbers) < 20:
            return 0.5
        tp = numbers[-1]
        sma = np.mean(numbers[-20:])
        mad = np.mean([abs(n - sma) for n in numbers[-20:]])
        if mad == 0:
            return 0.5
        cci = (tp - sma) / (0.015 * mad)
        return 0.5 + (cci / 400)
    
    def algo_adx(self, numbers):
        """Average Directional Index"""
        if len(numbers) < 14:
            return 0.5
        ups = [max(0, numbers[i] - numbers[i-1]) for i in range(1, len(numbers[-14:]))]
        downs = [max(0, numbers[i-1] - numbers[i]) for i in range(1, len(numbers[-14:]))]
        avg_up = np.mean(ups)
        avg_down = np.mean(downs)
        dx = abs(avg_up - avg_down) / (avg_up + avg_down + 0.001) * 100
        return dx / 100

    # ============================================
    # 📊 ALGORITHM 51-60: VOLATILITY ANALYSIS
    # ============================================
    
    def algo_bollinger_upper(self, numbers):
        """Bollinger Band Upper"""
        if len(numbers) < 20:
            return 0.5
        sma = np.mean(numbers[-20:])
        std = np.std(numbers[-20:])
        upper = sma + 2 * std
        return 1 if numbers[-1] > upper else 0.5
    
    def algo_bollinger_lower(self, numbers):
        """Bollinger Band Lower"""
        if len(numbers) < 20:
            return 0.5
        sma = np.mean(numbers[-20:])
        std = np.std(numbers[-20:])
        lower = sma - 2 * std
        return 0 if numbers[-1] < lower else 0.5
    
    def algo_atr(self, numbers):
        """Average True Range"""
        if len(numbers) < 14:
            return 0.5
        ranges = [abs(numbers[i] - numbers[i-1]) for i in range(1, len(numbers[-14:]))]
        atr = np.mean(ranges)
        return atr / 9
    
    def algo_volatility_10(self, numbers):
        """10-Period Volatility"""
        if len(numbers) < 10:
            return 0.5
        return np.std(numbers[-10:]) / 4.5
    
    def algo_volatility_20(self, numbers):
        """20-Period Volatility"""
        if len(numbers) < 20:
            return 0.5
        return np.std(numbers[-20:]) / 4.5
    
    def algo_range_analysis(self, numbers):
        """Range Analysis"""
        if len(numbers) < 10:
            return 0.5
        range_val = max(numbers[-10:]) - min(numbers[-10:])
        return range_val / 9
    
    def algo_volatility_ratio(self, numbers):
        """Volatility Ratio"""
        if len(numbers) < 20:
            return 0.5
        vol_short = np.std(numbers[-5:])
        vol_long = np.std(numbers[-20:])
        return vol_short / (vol_long + 0.001) / 2
    
    def algo_chaikin_volatility(self, numbers):
        """Chaikin Volatility"""
        if len(numbers) < 20:
            return 0.5
        hl_10 = max(numbers[-10:]) - min(numbers[-10:])
        hl_20 = max(numbers[-20:]) - min(numbers[-20:])
        cv = (hl_10 - hl_20) / (hl_20 + 0.001)
        return 0.5 + cv
    
    def algo_ulcer_index(self, numbers):
        """Ulcer Index"""
        if len(numbers) < 14:
            return 0.5
        max_val = max(numbers[-14:])
        drawdowns = [(max_val - n) / max_val * 100 for n in numbers[-14:]]
        ui = np.sqrt(np.mean([d**2 for d in drawdowns]))
        return ui / 100
    
    def algo_historical_volatility(self, numbers):
        """Historical Volatility"""
        if len(numbers) < 20:
            return 0.5
        returns = [(numbers[i] - numbers[i-1]) / (numbers[i-1] + 0.001) for i in range(1, len(numbers[-20:]))]
        hv = np.std(returns) * np.sqrt(252)
        return min(1, hv)

    # ============================================
    # 📊 ALGORITHM 61-70: TREND ANALYSIS
    # ============================================
    
    def algo_linear_regression(self, numbers):
        """Linear Regression Trend"""
        if len(numbers) < 10:
            return 0.5
        x = np.arange(len(numbers[-10:]))
        slope, intercept, _, _, _ = stats.linregress(x, numbers[-10:])
        predicted = slope * len(numbers[-10:]) + intercept
        return predicted / 9
    
    def algo_polynomial_trend(self, numbers):
        """Polynomial Trend"""
        if len(numbers) < 10:
            return 0.5
        x = np.arange(len(numbers[-10:]))
        coeffs = np.polyfit(x, numbers[-10:], 2)
        predicted = np.polyval(coeffs, len(numbers[-10:]))
        return max(0, min(1, predicted / 9))
    
    def algo_trend_strength(self, numbers):
        """Trend Strength"""
        if len(numbers) < 10:
            return 0.5
        ups = sum(1 for i in range(1, len(numbers[-10:])) if numbers[-10:][i] > numbers[-10:][i-1])
        return ups / 9
    
    def algo_trend_direction(self, numbers):
        """Trend Direction"""
        if len(numbers) < 5:
            return 0.5
        return 0.7 if numbers[-1] > numbers[-5] else 0.3
    
    def algo_parabolic_sar(self, numbers):
        """Parabolic SAR"""
        if len(numbers) < 10:
            return 0.5
        trend = 1 if numbers[-1] > numbers[-5] else -1
        return 0.7 if trend > 0 else 0.3
    
    def algo_supertrend(self, numbers):
        """Supertrend Indicator"""
        if len(numbers) < 14:
            return 0.5
        atr = np.mean([abs(numbers[i] - numbers[i-1]) for i in range(1, len(numbers[-14:]))])
        upper = np.mean(numbers[-14:]) + 2 * atr
        lower = np.mean(numbers[-14:]) - 2 * atr
        return 0.7 if numbers[-1] > lower else 0.3
    
    def algo_ichimoku(self, numbers):
        """Ichimoku Cloud"""
        if len(numbers) < 26:
            return 0.5
        tenkan = (max(numbers[-9:]) + min(numbers[-9:])) / 2
        kijun = (max(numbers[-26:]) + min(numbers[-26:])) / 2
        return 0.7 if tenkan > kijun else 0.3
    
    def algo_vwap(self, numbers):
        """VWAP-like Analysis"""
        if len(numbers) < 10:
            return 0.5
        weights = list(range(1, 11))
        vwap = np.average(numbers[-10:], weights=weights)
        return vwap / 9
    
    def algo_keltner_channel(self, numbers):
        """Keltner Channel"""
        if len(numbers) < 20:
            return 0.5
        ema = np.mean(numbers[-20:])
        atr = np.mean([abs(numbers[i] - numbers[i-1]) for i in range(1, len(numbers[-14:]))])
        upper = ema + 2 * atr
        lower = ema - 2 * atr
        if numbers[-1] > upper:
            return 0.8
        elif numbers[-1] < lower:
            return 0.2
        return 0.5
    
    def algo_donchian_channel(self, numbers):
        """Donchian Channel"""
        if len(numbers) < 20:
            return 0.5
        upper = max(numbers[-20:])
        lower = min(numbers[-20:])
        mid = (upper + lower) / 2
        return (numbers[-1] - lower) / (upper - lower + 0.001)

    # ============================================
    # 📊 ALGORITHM 71-80: MACHINE LEARNING
    # ============================================
    
    def algo_random_forest(self, numbers, results):
        """Random Forest Prediction"""
        if len(numbers) < 30:
            return 0.5
        try:
            X = []
            y = []
            for i in range(5, len(numbers)-1):
                X.append(numbers[i-5:i])
                y.append(1 if results[i] == 'BIG' else 0)
            if len(X) < 10:
                return 0.5
            clf = RandomForestClassifier(n_estimators=10, random_state=42, max_depth=3)
            clf.fit(X, y)
            prediction = clf.predict_proba([numbers[-5:]])[0]
            return prediction[1] if len(prediction) > 1 else 0.5
        except:
            return 0.5
    
    def algo_gradient_boosting(self, numbers, results):
        """Gradient Boosting Prediction"""
        if len(numbers) < 30:
            return 0.5
        try:
            X = []
            y = []
            for i in range(5, len(numbers)-1):
                X.append(numbers[i-5:i])
                y.append(1 if results[i] == 'BIG' else 0)
            if len(X) < 10:
                return 0.5
            clf = GradientBoostingClassifier(n_estimators=10, random_state=42, max_depth=2)
            clf.fit(X, y)
            prediction = clf.predict_proba([numbers[-5:]])[0]
            return prediction[1] if len(prediction) > 1 else 0.5
        except:
            return 0.5
    
    def algo_logistic_regression(self, numbers, results):
        """Logistic Regression Prediction"""
        if len(numbers) < 30:
            return 0.5
        try:
            X = []
            y = []
            for i in range(5, len(numbers)-1):
                X.append(numbers[i-5:i])
                y.append(1 if results[i] == 'BIG' else 0)
            if len(X) < 10:
                return 0.5
            clf = LogisticRegression(random_state=42, max_iter=100)
            clf.fit(X, y)
            prediction = clf.predict_proba([numbers[-5:]])[0]
            return prediction[1] if len(prediction) > 1 else 0.5
        except:
            return 0.5
    
    def algo_naive_bayes(self, numbers, results):
        """Naive Bayes Prediction"""
        if len(numbers) < 30:
            return 0.5
        try:
            X = []
            y = []
            for i in range(5, len(numbers)-1):
                X.append(numbers[i-5:i])
                y.append(1 if results[i] == 'BIG' else 0)
            if len(X) < 10:
                return 0.5
            clf = GaussianNB()
            clf.fit(X, y)
            prediction = clf.predict_proba([numbers[-5:]])[0]
            return prediction[1] if len(prediction) > 1 else 0.5
        except:
            return 0.5
    
    def algo_decision_tree(self, numbers, results):
        """Decision Tree Prediction"""
        if len(numbers) < 30:
            return 0.5
        try:
            X = []
            y = []
            for i in range(5, len(numbers)-1):
                X.append(numbers[i-5:i])
                y.append(1 if results[i] == 'BIG' else 0)
            if len(X) < 10:
                return 0.5
            clf = DecisionTreeClassifier(random_state=42, max_depth=5)
            clf.fit(X, y)
            prediction = clf.predict_proba([numbers[-5:]])[0]
            return prediction[1] if len(prediction) > 1 else 0.5
        except:
            return 0.5
    
    def algo_knn_3(self, numbers, results):
        """K-Nearest Neighbors (k=3)"""
        if len(numbers) < 30:
            return 0.5
        try:
            X = []
            y = []
            for i in range(5, len(numbers)-1):
                X.append(numbers[i-5:i])
                y.append(1 if results[i] == 'BIG' else 0)
            if len(X) < 10:
                return 0.5
            clf = KNeighborsClassifier(n_neighbors=3)
            clf.fit(X, y)
            prediction = clf.predict_proba([numbers[-5:]])[0]
            return prediction[1] if len(prediction) > 1 else 0.5
        except:
            return 0.5
    
    def algo_knn_5(self, numbers, results):
        """K-Nearest Neighbors (k=5)"""
        if len(numbers) < 30:
            return 0.5
        try:
            X = []
            y = []
            for i in range(5, len(numbers)-1):
                X.append(numbers[i-5:i])
                y.append(1 if results[i] == 'BIG' else 0)
            if len(X) < 10:
                return 0.5
            clf = KNeighborsClassifier(n_neighbors=5)
            clf.fit(X, y)
            prediction = clf.predict_proba([numbers[-5:]])[0]
            return prediction[1] if len(prediction) > 1 else 0.5
        except:
            return 0.5
    
    def algo_knn_7(self, numbers, results):
        """K-Nearest Neighbors (k=7)"""
        if len(numbers) < 30:
            return 0.5
        try:
            X = []
            y = []
            for i in range(5, len(numbers)-1):
                X.append(numbers[i-5:i])
                y.append(1 if results[i] == 'BIG' else 0)
            if len(X) < 10:
                return 0.5
            clf = KNeighborsClassifier(n_neighbors=7)
            clf.fit(X, y)
            prediction = clf.predict_proba([numbers[-5:]])[0]
            return prediction[1] if len(prediction) > 1 else 0.5
        except:
            return 0.5
    
    def algo_svm_linear(self, numbers, results):
        """SVM Linear Kernel"""
        if len(numbers) < 30:
            return 0.5
        try:
            X = []
            y = []
            for i in range(5, len(numbers)-1):
                X.append(numbers[i-5:i])
                y.append(1 if results[i] == 'BIG' else 0)
            if len(X) < 10:
                return 0.5
            clf = SVC(kernel='linear', probability=True, random_state=42)
            clf.fit(X, y)
            prediction = clf.predict_proba([numbers[-5:]])[0]
            return prediction[1] if len(prediction) > 1 else 0.5
        except:
            return 0.5
    
    def algo_svm_rbf(self, numbers, results):
        """SVM RBF Kernel"""
        if len(numbers) < 30:
            return 0.5
        try:
            X = []
            y = []
            for i in range(5, len(numbers)-1):
                X.append(numbers[i-5:i])
                y.append(1 if results[i] == 'BIG' else 0)
            if len(X) < 10:
                return 0.5
            clf = SVC(kernel='rbf', probability=True, random_state=42)
            clf.fit(X, y)
            prediction = clf.predict_proba([numbers[-5:]])[0]
            return prediction[1] if len(prediction) > 1 else 0.5
        except:
            return 0.5

    # ============================================
    # 📊 ALGORITHM 81-90: MARKOV & PROBABILITY
    # ============================================
    
    def algo_markov_chain_1(self, results):
        """First Order Markov Chain"""
        if len(results) < 20:
            return 0.5
        transitions = {'BIG': {'BIG': 0, 'SMALL': 0}, 'SMALL': {'BIG': 0, 'SMALL': 0}}
        for i in range(1, len(results)):
            transitions[results[i-1]][results[i]] += 1
        last = results[-1]
        total = transitions[last]['BIG'] + transitions[last]['SMALL']
        if total == 0:
            return 0.5
        return transitions[last]['BIG'] / total
    
    def algo_markov_chain_2(self, results):
        """Second Order Markov Chain"""
        if len(results) < 30:
            return 0.5
        transitions = {}
        for i in range(2, len(results)):
            state = (results[i-2], results[i-1])
            if state not in transitions:
                transitions[state] = {'BIG': 0, 'SMALL': 0}
            transitions[state][results[i]] += 1
        last_state = (results[-2], results[-1])
        if last_state not in transitions:
            return 0.5
        total = transitions[last_state]['BIG'] + transitions[last_state]['SMALL']
        if total == 0:
            return 0.5
        return transitions[last_state]['BIG'] / total
    
    def algo_markov_chain_3(self, results):
        """Third Order Markov Chain"""
        if len(results) < 40:
            return 0.5
        transitions = {}
        for i in range(3, len(results)):
            state = (results[i-3], results[i-2], results[i-1])
            if state not in transitions:
                transitions[state] = {'BIG': 0, 'SMALL': 0}
            transitions[state][results[i]] += 1
        last_state = (results[-3], results[-2], results[-1])
        if last_state not in transitions:
            return 0.5
        total = transitions[last_state]['BIG'] + transitions[last_state]['SMALL']
        if total == 0:
            return 0.5
        return transitions[last_state]['BIG'] / total
    
    def algo_bayesian_1(self, results):
        """Bayesian Probability 1"""
        if len(results) < 20:
            return 0.5
        big_count = sum(1 for r in results[-20:] if r == 'BIG')
        prior_big = big_count / 20
        # Apply Laplace smoothing
        return (big_count + 1) / (20 + 2)
    
    def algo_bayesian_2(self, results):
        """Bayesian with Streak Prior"""
        if len(results) < 10:
            return 0.5
        streak = 1
        for i in range(len(results)-2, -1, -1):
            if results[i] == results[-1]:
                streak += 1
            else:
                break
        # Higher probability of opposite after streak
        if streak >= 3:
            return 0.3 if results[-1] == 'BIG' else 0.7
        return 0.5
    
    def algo_conditional_prob(self, results):
        """Conditional Probability"""
        if len(results) < 30:
            return 0.5
        last = results[-1]
        same_outcomes = [results[i+1] for i in range(len(results)-1) if results[i] == last]
        if not same_outcomes:
            return 0.5
        big_after = sum(1 for r in same_outcomes if r == 'BIG')
        return big_after / len(same_outcomes)
    
    def algo_joint_probability(self, results):
        """Joint Probability Analysis"""
        if len(results) < 30:
            return 0.5
        pattern = (results[-2], results[-1])
        matches = []
        for i in range(2, len(results)-1):
            if (results[i-2], results[i-1]) == pattern:
                matches.append(results[i])
        if not matches:
            return 0.5
        return sum(1 for m in matches if m == 'BIG') / len(matches)
    
    def algo_likelihood_ratio(self, results):
        """Likelihood Ratio Analysis"""
        if len(results) < 20:
            return 0.5
        big_count = sum(1 for r in results[-20:] if r == 'BIG')
        small_count = 20 - big_count
        if small_count == 0:
            return 0.9
        lr = big_count / (small_count + 0.001)
        return min(0.9, max(0.1, lr / 2))
    
    def algo_posterior_probability(self, results):
        """Posterior Probability"""
        if len(results) < 20:
            return 0.5
        prior = 0.5
        likelihood = sum(1 for r in results[-10:] if r == 'BIG') / 10
        evidence = sum(1 for r in results[-20:] if r == 'BIG') / 20
        if evidence == 0:
            return 0.5
        posterior = (likelihood * prior) / evidence
        return min(0.9, max(0.1, posterior))
    
    def algo_monte_carlo(self, results):
        """Monte Carlo Simulation"""
        if len(results) < 20:
            return 0.5
        simulations = 100
        big_wins = 0
        prob_big = sum(1 for r in results[-20:] if r == 'BIG') / 20
        for _ in range(simulations):
            if np.random.random() < prob_big:
                big_wins += 1
        return big_wins / simulations

    # ============================================
    # 📊 ALGORITHM 91-100: ADVANCED ENSEMBLE
    # ============================================
    
    def algo_weighted_voting(self, scores):
        """Weighted Voting Ensemble"""
        if not scores:
            return 0.5
        weights = np.exp(np.linspace(0, 1, len(scores)))
        return np.average(scores, weights=weights)
    
    def algo_majority_voting(self, scores):
        """Majority Voting"""
        if not scores:
            return 0.5
        big_votes = sum(1 for s in scores if s > 0.5)
        return big_votes / len(scores)
    
    def algo_soft_voting(self, scores):
        """Soft Voting Average"""
        if not scores:
            return 0.5
        return np.mean(scores)
    
    def algo_hard_voting(self, scores):
        """Hard Voting"""
        if not scores:
            return 0.5
        predictions = [1 if s > 0.5 else 0 for s in scores]
        return np.mean(predictions)
    
    def algo_stacking(self, scores):
        """Stacking Ensemble"""
        if len(scores) < 5:
            return 0.5
        top_5 = sorted(scores, reverse=True)[:5]
        return np.mean(top_5)
    
    def algo_bagging(self, scores):
        """Bagging Ensemble"""
        if not scores:
            return 0.5
        bootstrap_samples = [np.random.choice(scores, size=len(scores)//2) for _ in range(10)]
        bootstrap_means = [np.mean(s) for s in bootstrap_samples]
        return np.mean(bootstrap_means)
    
    def algo_boosting_ensemble(self, scores):
        """Boosting-style Ensemble"""
        if not scores:
            return 0.5
        weights = np.ones(len(scores)) / len(scores)
        for i, s in enumerate(scores):
            error = abs(0.5 - s)
            weights[i] *= np.exp(error)
        weights /= weights.sum()
        return np.average(scores, weights=weights)
    
    def algo_confidence_weighted(self, scores):
        """Confidence Weighted Ensemble"""
        if not scores:
            return 0.5
        confidences = [abs(s - 0.5) for s in scores]
        if sum(confidences) == 0:
            return 0.5
        return np.average(scores, weights=confidences)
    
    def algo_dynamic_ensemble(self, scores):
        """Dynamic Ensemble Selection"""
        if not scores:
            return 0.5
        # Select top confident predictions
        sorted_scores = sorted(scores, key=lambda x: abs(x - 0.5), reverse=True)
        top_confident = sorted_scores[:len(scores)//3 + 1]
        return np.mean(top_confident)
    
    def algo_adaptive_ensemble(self, scores):
        """Adaptive Ensemble"""
        if not scores:
            return 0.5
        # Weight recent algorithms more
        recent_weights = np.exp(np.linspace(-1, 0, len(scores)))
        return np.average(scores, weights=recent_weights)

    # ============================================
    # 🎯 MASTER PREDICTION ENGINE
    # ============================================
    
    def get_master_prediction(self, data):
        """Master prediction combining all 100+ algorithms"""
        extracted = self.extract_results(data)
        
        if not extracted or len(extracted) < 5:
            return {
                "success": False,
                "error": "Not enough data for prediction",
                "data_count": len(extracted) if extracted else 0
            }
        
        numbers = [r['number'] for r in extracted]
        results = [r['result'] for r in extracted]
        current_period = extracted[0]['period']
        
        # Check if we have a previous prediction to evaluate
        win_loss_status = None
        if self.last_prediction and self.last_period:
            # Find if last period result is available
            for r in extracted:
                if str(r['period']) == str(self.last_period):
                    actual_result = r['result']
                    if actual_result == self.last_prediction:
                        win_loss_status = "WIN ✅"
                        self.win_count += 1
                    else:
                        win_loss_status = "LOSS ❌"
                        self.loss_count += 1
                    break
        
        all_scores = []
        algorithm_results = {}
        
        # Algorithms 1-10: Moving Averages
        algorithm_results['SMA_3'] = self.algo_sma_3(numbers)
        algorithm_results['SMA_5'] = self.algo_sma_5(numbers)
        algorithm_results['SMA_7'] = self.algo_sma_7(numbers)
        algorithm_results['SMA_10'] = self.algo_sma_10(numbers)
        algorithm_results['SMA_15'] = self.algo_sma_15(numbers)
        algorithm_results['EMA_3'] = self.algo_ema_3(numbers)
        algorithm_results['EMA_5'] = self.algo_ema_5(numbers)
        algorithm_results['EMA_7'] = self.algo_ema_7(numbers)
        algorithm_results['WMA_5'] = self.algo_wma_5(numbers)
        algorithm_results['WMA_10'] = self.algo_wma_10(numbers)
        
        # Algorithms 11-20: Statistical
        algorithm_results['Mean_Reversion'] = self.algo_mean_reversion(numbers)
        algorithm_results['Std_Deviation'] = self.algo_std_deviation(numbers)
        algorithm_results['Variance'] = self.algo_variance_analysis(numbers)
        algorithm_results['Skewness'] = self.algo_skewness(numbers)
        algorithm_results['Kurtosis'] = self.algo_kurtosis(numbers)
        algorithm_results['Median'] = self.algo_median_analysis(numbers)
        algorithm_results['Mode'] = self.algo_mode_analysis(numbers)
        algorithm_results['Percentile_25'] = self.algo_percentile_25(numbers)
        algorithm_results['Percentile_75'] = self.algo_percentile_75(numbers)
        algorithm_results['IQR'] = self.algo_iqr_analysis(numbers)
        
        # Algorithms 21-30: Pattern Recognition
        algorithm_results['Streak'] = self.algo_streak_analysis(results)
        algorithm_results['Alternating'] = self.algo_alternating_pattern(results)
        algorithm_results['Triple_Pattern'] = self.algo_triple_pattern(results)
        algorithm_results['Double_Pattern'] = self.algo_double_pattern(results)
        algorithm_results['Pattern_5'] = self.algo_pattern_5(results)
        algorithm_results['Fibonacci'] = self.algo_fibonacci_pattern(numbers)
        algorithm_results['Prime'] = self.algo_prime_pattern(numbers)
        algorithm_results['Even_Odd'] = self.algo_even_odd_pattern(numbers)
        algorithm_results['High_Low'] = self.algo_high_low_pattern(numbers)
        algorithm_results['Gap'] = self.algo_gap_analysis(numbers)
        
        # Algorithms 31-40: Frequency
        algorithm_results['Freq_Small'] = self.algo_frequency_small(results)
        algorithm_results['Freq_Big'] = self.algo_frequency_big(results)
        algorithm_results['Hot_Numbers'] = self.algo_hot_numbers(numbers)
        algorithm_results['Cold_Numbers'] = self.algo_cold_numbers(numbers)
        algorithm_results['Due_Numbers'] = self.algo_due_numbers(numbers)
        algorithm_results['Repeat_Freq'] = self.algo_repeat_frequency(numbers)
        algorithm_results['Cluster'] = self.algo_cluster_analysis(numbers)
        algorithm_results['Chi2'] = self.algo_distribution_chi2(numbers)
        algorithm_results['Entropy'] = self.algo_entropy_analysis(numbers)
        algorithm_results['Gini'] = self.algo_gini_coefficient(numbers)
        
        # Algorithms 41-50: Momentum
        algorithm_results['RSI'] = self.algo_rsi(numbers)
        algorithm_results['MACD'] = self.algo_macd(numbers)
        algorithm_results['Stochastic'] = self.algo_stochastic(numbers)
        algorithm_results['Momentum_3'] = self.algo_momentum_3(numbers)
        algorithm_results['Momentum_5'] = self.algo_momentum_5(numbers)
        algorithm_results['Momentum_10'] = self.algo_momentum_10(numbers)
        algorithm_results['ROC'] = self.algo_roc(numbers)
        algorithm_results['Williams_R'] = self.algo_williams_r(numbers)
        algorithm_results['CCI'] = self.algo_cci(numbers)
        algorithm_results['ADX'] = self.algo_adx(numbers)
        
        # Algorithms 51-60: Volatility
        algorithm_results['Bollinger_Upper'] = self.algo_bollinger_upper(numbers)
        algorithm_results['Bollinger_Lower'] = self.algo_bollinger_lower(numbers)
        algorithm_results['ATR'] = self.algo_atr(numbers)
        algorithm_results['Volatility_10'] = self.algo_volatility_10(numbers)
        algorithm_results['Volatility_20'] = self.algo_volatility_20(numbers)
        algorithm_results['Range'] = self.algo_range_analysis(numbers)
        algorithm_results['Volatility_Ratio'] = self.algo_volatility_ratio(numbers)
        algorithm_results['Chaikin_Vol'] = self.algo_chaikin_volatility(numbers)
        algorithm_results['Ulcer_Index'] = self.algo_ulcer_index(numbers)
        algorithm_results['Historical_Vol'] = self.algo_historical_volatility(numbers)
        
        # Algorithms 61-70: Trend
        algorithm_results['Linear_Reg'] = self.algo_linear_regression(numbers)
        algorithm_results['Polynomial'] = self.algo_polynomial_trend(numbers)
        algorithm_results['Trend_Strength'] = self.algo_trend_strength(numbers)
        algorithm_results['Trend_Direction'] = self.algo_trend_direction(numbers)
        algorithm_results['Parabolic_SAR'] = self.algo_parabolic_sar(numbers)
        algorithm_results['Supertrend'] = self.algo_supertrend(numbers)
        algorithm_results['Ichimoku'] = self.algo_ichimoku(numbers)
        algorithm_results['VWAP'] = self.algo_vwap(numbers)
        algorithm_results['Keltner'] = self.algo_keltner_channel(numbers)
        algorithm_results['Donchian'] = self.algo_donchian_channel(numbers)
        
        # Algorithms 71-80: Machine Learning
        algorithm_results['Random_Forest'] = self.algo_random_forest(numbers, results)
        algorithm_results['Gradient_Boost'] = self.algo_gradient_boosting(numbers, results)
        algorithm_results['Logistic_Reg'] = self.algo_logistic_regression(numbers, results)
        algorithm_results['Naive_Bayes'] = self.algo_naive_bayes(numbers, results)
        algorithm_results['Decision_Tree'] = self.algo_decision_tree(numbers, results)
        algorithm_results['KNN_3'] = self.algo_knn_3(numbers, results)
        algorithm_results['KNN_5'] = self.algo_knn_5(numbers, results)
        algorithm_results['KNN_7'] = self.algo_knn_7(numbers, results)
        algorithm_results['SVM_Linear'] = self.algo_svm_linear(numbers, results)
        algorithm_results['SVM_RBF'] = self.algo_svm_rbf(numbers, results)
        
        # Algorithms 81-90: Markov & Probability
        algorithm_results['Markov_1'] = self.algo_markov_chain_1(results)
        algorithm_results['Markov_2'] = self.algo_markov_chain_2(results)
        algorithm_results['Markov_3'] = self.algo_markov_chain_3(results)
        algorithm_results['Bayesian_1'] = self.algo_bayesian_1(results)
        algorithm_results['Bayesian_2'] = self.algo_bayesian_2(results)
        algorithm_results['Conditional'] = self.algo_conditional_prob(results)
        algorithm_results['Joint_Prob'] = self.algo_joint_probability(results)
        algorithm_results['Likelihood'] = self.algo_likelihood_ratio(results)
        algorithm_results['Posterior'] = self.algo_posterior_probability(results)
        algorithm_results['Monte_Carlo'] = self.algo_monte_carlo(results)
        
        # Collect all scores
        all_scores = list(algorithm_results.values())
        
        # Algorithms 91-100: Ensemble Methods
        algorithm_results['Weighted_Vote'] = self.algo_weighted_voting(all_scores)
        algorithm_results['Majority_Vote'] = self.algo_majority_voting(all_scores)
        algorithm_results['Soft_Vote'] = self.algo_soft_voting(all_scores)
        algorithm_results['Hard_Vote'] = self.algo_hard_voting(all_scores)
        algorithm_results['Stacking'] = self.algo_stacking(all_scores)
        algorithm_results['Bagging'] = self.algo_bagging(all_scores)
        algorithm_results['Boosting'] = self.algo_boosting_ensemble(all_scores)
        algorithm_results['Confidence_Weighted'] = self.algo_confidence_weighted(all_scores)
        algorithm_results['Dynamic_Ensemble'] = self.algo_dynamic_ensemble(all_scores)
        algorithm_results['Adaptive_Ensemble'] = self.algo_adaptive_ensemble(all_scores)
        
        # Final Master Score
        final_scores = list(algorithm_results.values())
        
        # Advanced Final Calculation
        mean_score = np.mean(final_scores)
        median_score = np.median(final_scores)
        weighted_score = np.average(final_scores, weights=np.exp(np.linspace(0, 1, len(final_scores))))
        
        # Confidence weighted by deviation from 0.5
        confidences = [abs(s - 0.5) * 2 for s in final_scores]
        if sum(confidences) > 0:
            confidence_weighted = np.average(final_scores, weights=confidences)
        else:
            confidence_weighted = 0.5
        
        # Master Score
        master_score = (mean_score * 0.25 + median_score * 0.25 + 
                       weighted_score * 0.25 + confidence_weighted * 0.25)
        
        # Determine prediction
        prediction = "BIG" if master_score >= 0.5 else "SMALL"
        confidence = abs(master_score - 0.5) * 200  # 0-100%
        
        # Calculate next period
        try:
            next_period = str(int(current_period) + 1)
        except:
            next_period = f"{current_period}_next"
        
        # Store prediction for next comparison
        self.last_prediction = prediction
        self.last_period = next_period
        
        # Algorithm votes count
        big_votes = sum(1 for s in final_scores if s >= 0.5)
        small_votes = len(final_scores) - big_votes
        
        return {
            "success": True,
            "timestamp": datetime.now().isoformat(),
            "current_period": current_period,
            "next_period": next_period,
            "last_result": {
                "period": extracted[0]['period'],
                "number": extracted[0]['number'],
                "result": extracted[0]['result']
            },
            "prediction": {
                "value": prediction,
                "confidence": round(confidence, 2),
                "master_score": round(master_score, 4)
            },
            "previous_prediction_result": win_loss_status,
            "statistics": {
                "total_predictions": self.win_count + self.loss_count,
                "wins": self.win_count,
                "losses": self.loss_count,
                "win_rate": round(self.win_count / (self.win_count + self.loss_count) * 100, 2) if (self.win_count + self.loss_count) > 0 else 0
            },
            "algorithm_analysis": {
                "total_algorithms": len(algorithm_results),
                "big_votes": big_votes,
                "small_votes": small_votes,
                "mean_score": round(mean_score, 4),
                "median_score": round(median_score, 4),
                "weighted_score": round(weighted_score, 4),
                "confidence_weighted": round(confidence_weighted, 4)
            },
            "top_5_confident_algorithms": dict(sorted(
                algorithm_results.items(), 
                key=lambda x: abs(x[1] - 0.5), 
                reverse=True
            )[:5]),
            "recent_history": extracted[:5]
        }


# Global engine instance
engine = UltraPredictionEngine()

class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        try:
            # Fetch data and get prediction
            data = engine.fetch_data()
            
            if "error" in data:
                result = {
                    "success": False,
                    "error": f"Failed to fetch data: {data['error']}"
                }
            else:
                result = engine.get_master_prediction(data)
            
            # Send response
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.send_header('Cache-Control', 'no-cache, no-store, must-revalidate')
            self.end_headers()
            
            response = json.dumps(result, indent=2, ensure_ascii=False)
            self.wfile.write(response.encode())
            
        except Exception as e:
            self.send_response(500)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            error_response = json.dumps({
                "success": False,
                "error": str(e)
            })
            self.wfile.write(error_response.encode())
    
    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()