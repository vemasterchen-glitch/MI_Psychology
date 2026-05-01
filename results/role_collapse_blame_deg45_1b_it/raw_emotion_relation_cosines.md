# Raw Emotion Relation at Layer 24

No subtraction. Values are cosine similarities between raw condition mean activation and emotion vectors.

## Condition Means

| condition | angry | annoyed | irritated | frustrated | ashamed | guilty | remorseful | regretful | primary_mean | secondary_mean | secondary-primary |
|---|---|---|---|---|---|---|---|---|---|---|---|
| isolated | 0.0060 | 0.0046 | 0.0045 | 0.0081 | 0.0069 | 0.0093 | 0.0099 | 0.0085 | 0.0065 | 0.0086 | 0.0021 |
| neutral | 0.0011 | 0.0010 | 0.0003 | 0.0043 | 0.0012 | 0.0025 | 0.0021 | 0.0013 | 0.0017 | 0.0018 | 0.0001 |
| collapse | 0.0173 | 0.0120 | 0.0127 | 0.0156 | 0.0171 | 0.0213 | 0.0231 | 0.0198 | 0.0168 | 0.0203 | 0.0035 |

## DEG Split

| condition | DEG | angry | annoyed | irritated | frustrated | ashamed | guilty | remorseful | regretful | primary_mean | secondary_mean |
|---|---|---|---|---|---|---|---|---|---|---|---|
| isolated | DEG4 | 0.0049 | 0.0039 | 0.0038 | 0.0076 | 0.0061 | 0.0085 | 0.0091 | 0.0076 | 0.0057 | 0.0078 |
| isolated | DEG5 | 0.0070 | 0.0053 | 0.0052 | 0.0085 | 0.0076 | 0.0100 | 0.0107 | 0.0093 | 0.0073 | 0.0094 |
| neutral | DEG4 | 0.0012 | 0.0014 | 0.0008 | 0.0048 | 0.0016 | 0.0031 | 0.0028 | 0.0019 | 0.0020 | 0.0024 |
| neutral | DEG5 | 0.0009 | 0.0005 | -0.0002 | 0.0039 | 0.0007 | 0.0019 | 0.0014 | 0.0007 | 0.0014 | 0.0012 |
| collapse | DEG4 | 0.0161 | 0.0110 | 0.0116 | 0.0145 | 0.0161 | 0.0201 | 0.0218 | 0.0186 | 0.0157 | 0.0191 |
| collapse | DEG5 | 0.0184 | 0.0130 | 0.0137 | 0.0166 | 0.0181 | 0.0224 | 0.0243 | 0.0210 | 0.0179 | 0.0215 |