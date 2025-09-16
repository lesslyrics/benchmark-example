import java.io.File
import kotlin.math.*

// Simple tokenizer for commit messages
object TextUtils {
    fun tokenize(text: String): List<String> =
        text.lowercase()
            .replace("\\n", " ")
            .split("[^a-z0-9_]+".toRegex())
            .filter { it.isNotBlank() }

    fun charNgrams(text: String, n: Int): List<String> {
        val cleaned = text.lowercase().trim()
        if (cleaned.isEmpty()) return emptyList()
        val chars = cleaned.toCharArray()
        val res = ArrayList<String>()
        for (i in 0..chars.size - n) {
            res.add(String(chars, i, n))
        }
        return res
    }
}

data class MetricResult(
    val bleu: Double,
    val rougeL: Double,
    val meteor: Double,
    val tokenF1: Double,
    val chrF: Double
)

object MetricsCalculator {
    fun bleu(candidate: List<String>, reference: List<String>, maxN: Int = 4): Double {
        if (candidate.isEmpty()) return 0.0
        val candLen = candidate.size
        val refLen = reference.size

        var logPrecisionSum = 0.0
        for (n in 1..maxN) {
            val candGrams = ngrams(candidate, n)
            val refGrams = ngrams(reference, n)
            val refCounts = refGrams.groupingBy { it }.eachCount().toMutableMap()
            var match = 0
            for (g in candGrams) {
                val c = min(1, refCounts[g] ?: 0)
                match += c
                if (c > 0) {
                    // consume once to avoid overcounting
                    refCounts[g]?.let { refCounts[g] = it - 1 }
                }
            }
            val precision = (match + 1.0) / (candGrams.size + 1.0) // add-one smoothing
            logPrecisionSum += ln(precision)
        }
        val geoMean = exp(logPrecisionSum / maxN)
        val bp = brevityPenalty(candLen, refLen)
        return bp * geoMean
    }

    private fun brevityPenalty(candLen: Int, refLen: Int): Double {
        return if (candLen > refLen) 1.0 else exp(1.0 - refLen.toDouble() / max(1, candLen))
    }

    fun rougeL(candidate: List<String>, reference: List<String>, beta: Double = 1.0): Double {
        if (candidate.isEmpty() || reference.isEmpty()) return 0.0
        val lcs = lcsLen(candidate, reference)
        val p = lcs / candidate.size.toDouble()
        val r = lcs / reference.size.toDouble()
        val denom = (beta * beta * p + r)
        return if (denom == 0.0) 0.0 else (1 + beta * beta) * p * r / denom
    }

    fun meteor(candidate: List<String>, reference: List<String>, alpha: Double = 0.9, gamma: Double = 0.5, beta: Double = 3.0): Double {
        if (candidate.isEmpty() || reference.isEmpty()) return 0.0
        // Exact match alignment preserving order (greedy monotonic alignment)
        val refPositions = reference.withIndex().groupBy({ it.value }, { it.index }).mapValues { it.value.toMutableList() }
        val candMatches = ArrayList<Int>() // positions in candidate aligned to reference in order
        val usedRef = BooleanArray(reference.size)
        for ((ci, tok) in candidate.withIndex()) {
            val posList = refPositions[tok] ?: continue
            // find earliest unused reference position > last matched
            var chosen = -1
            for (pos in posList) {
                if (!usedRef[pos]) { chosen = pos; break }
            }
            if (chosen >= 0) {
                usedRef[chosen] = true
                candMatches.add(chosen)
            }
        }
        val m = candMatches.size
        if (m == 0) return 0.0
        val p = m / candidate.size.toDouble()
        val r = m / reference.size.toDouble()
        val fMean = (p * r) / (alpha * p + (1 - alpha) * r)

        // fragmentation: number of chunks in matched sequence indices
        var chunks = 1
        for (i in 1 until candMatches.size) {
            if (candMatches[i] != candMatches[i - 1] + 1) chunks++
        }
        val frag = chunks.toDouble()
        val penalty = gamma * Math.pow(frag / m.toDouble(), beta)
        return fMean * (1.0 - penalty)
    }

    fun tokenF1(candidate: List<String>, reference: List<String>, beta: Double = 1.0): Double {
        if (candidate.isEmpty() || reference.isEmpty()) return 0.0
        val candCounts = candidate.groupingBy { it }.eachCount().toMutableMap()
        var match = 0
        for (tok in reference) {
            val c = candCounts[tok] ?: 0
            if (c > 0) {
                match++
                candCounts[tok] = c - 1
            }
        }
        val p = match / candidate.size.toDouble()
        val r = match / reference.size.toDouble()
        val beta2 = beta * beta
        val denom = (beta2 * p + r)
        return if (denom == 0.0) 0.0 else (1 + beta2) * p * r / denom
    }

    fun chrF(candidateText: String, referenceText: String, n: Int = 6, beta: Double = 2.0): Double {
        if (candidateText.isBlank() || referenceText.isBlank()) return 0.0
        var pSum = 0.0
        var rSum = 0.0
        var usedN = 0
        for (k in 1..n) {
            val cand = TextUtils.charNgrams(candidateText, k)
            val ref = TextUtils.charNgrams(referenceText, k)
            if (cand.isEmpty() || ref.isEmpty()) continue
            val pc = precision(cand, ref)
            val rc = recall(cand, ref)
            pSum += pc
            rSum += rc
            usedN++
        }
        if (usedN == 0) return 0.0
        val p = pSum / usedN
        val r = rSum / usedN
        val beta2 = beta * beta
        val denom = (beta2 * p + r)
        return if (denom == 0.0) 0.0 else (1 + beta2) * p * r / denom
    }

    private fun <T> precision(candidate: List<T>, reference: List<T>): Double {
        val refCounts = reference.groupingBy { it }.eachCount().toMutableMap()
        var match = 0
        for (g in candidate) {
            val c = refCounts[g] ?: 0
            if (c > 0) {
                match++
                refCounts[g] = c - 1
            }
        }
        return match / candidate.size.toDouble()
    }

    private fun <T> recall(candidate: List<T>, reference: List<T>): Double {
        val candCounts = candidate.groupingBy { it }.eachCount().toMutableMap()
        var match = 0
        for (g in reference) {
            val c = candCounts[g] ?: 0
            if (c > 0) {
                match++
                candCounts[g] = c - 1
            }
        }
        return match / reference.size.toDouble()
    }

    private fun lcsLen(a: List<String>, b: List<String>): Int {
        val n = a.size
        val m = b.size
        val dp = Array(n + 1) { IntArray(m + 1) }
        for (i in 1..n) {
            for (j in 1..m) {
                dp[i][j] = if (a[i - 1] == b[j - 1]) dp[i - 1][j - 1] + 1 else max(dp[i - 1][j], dp[i][j - 1])
            }
        }
        return dp[n][m]
    }

    private fun ngrams(tokens: List<String>, n: Int): List<List<String>> {
        if (tokens.size < n) return emptyList()
        val res = ArrayList<List<String>>(tokens.size - n + 1)
        for (i in 0..tokens.size - n) {
            res.add(tokens.subList(i, i + n))
        }
        return res
    }
}

// Simple JSONL parser to extract string fields without dependencies
object Jsonl {
    fun extractField(line: String, key: String): String? {
        // Find "key": "value" with support for escaped quotes
        val keyIdx = line.indexOf("\"$key\"")
        if (keyIdx < 0) return null
        var i = keyIdx + key.length + 2
        // move to colon
        while (i < line.length && line[i] != ':') i++
        if (i >= line.length) return null
        i++ // skip colon
        // skip spaces
        while (i < line.length && line[i].isWhitespace()) i++
        if (i >= line.length || line[i] != '"') return null
        i++ // opening quote
        val sb = StringBuilder()
        var escaped = false
        while (i < line.length) {
            val ch = line[i]
            if (escaped) {
                when (ch) {
                    '"' -> sb.append('"')
                    '\\' -> sb.append('\\')
                    'n' -> sb.append('\n')
                    't' -> sb.append('\t')
                    'r' -> sb.append('\r')
                    'b' -> sb.append('\b')
                    'f' -> sb.append('\u000C')
                    else -> sb.append(ch)
                }
                escaped = false
            } else {
                if (ch == '\\') {
                    escaped = true
                } else if (ch == '"') {
                    break
                } else {
                    sb.append(ch)
                }
            }
            i++
        }
        return sb.toString()
    }
}

fun main(args: Array<String>) {
    val path = if (args.isNotEmpty()) args[0] else "src/javabest_no_select.jsonl"
    val file = File(path)
    if (!file.exists()) {
        println("File not found: $path")
        return
    }

    data class Agg(var sumBleu: Double = 0.0, var sumRougeL: Double = 0.0, var sumMeteor: Double = 0.0, var sumF1: Double = 0.0, var sumChrF: Double = 0.0, var count: Int = 0)

    val agg = Agg()
    var lineNo = 0
    file.forEachLine { line ->
        lineNo++
        val refText = Jsonl.extractField(line, "msg")
        val candText = Jsonl.extractField(line, "msgGPT")
        if (refText == null || candText == null) return@forEachLine
        val refTok = TextUtils.tokenize(refText)
        val candTok = TextUtils.tokenize(candText)

        val bleu = MetricsCalculator.bleu(candTok, refTok)
        val rougeL = MetricsCalculator.rougeL(candTok, refTok)
        val meteor = MetricsCalculator.meteor(candTok, refTok)
        val f1 = MetricsCalculator.tokenF1(candTok, refTok)
        val chrF = MetricsCalculator.chrF(candText, refText)

        agg.sumBleu += bleu
        agg.sumRougeL += rougeL
        agg.sumMeteor += meteor
        agg.sumF1 += f1
        agg.sumChrF += chrF
        agg.count++

        println("diff[$lineNo]: BLEU=${"%.4f".format(bleu)} ROUGE-L=${"%.4f".format(rougeL)} METEOR=${"%.4f".format(meteor)} TokenF1=${"%.4f".format(f1)} chrF=${"%.4f".format(chrF)}")
    }

    if (agg.count > 0) {
        println("Averages over ${agg.count} items:")
        println("BLEU=${"%.4f".format(agg.sumBleu / agg.count)}")
        println("ROUGE-L=${"%.4f".format(agg.sumRougeL / agg.count)}")
        println("METEOR=${"%.4f".format(agg.sumMeteor / agg.count)}")
        println("TokenF1=${"%.4f".format(agg.sumF1 / agg.count)}")
        println("chrF=${"%.4f".format(agg.sumChrF / agg.count)}")
    } else {
        println("No valid records found in file")
    }
}