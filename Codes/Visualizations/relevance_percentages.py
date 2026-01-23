import matplotlib.pyplot as plt
import numpy as np


insurance_means, insurance_std = (0.0740, 0.1127, 0.0176, 0.1061, 0.1404, 0.0264), (0.1322, 0.1708, 0.0579, 0.1583, 0.1247, 0.0955)
insurance_unknown_means, insurance_unknown_std = (0.0028, 0.0115, 0.0000, 0.0013, 0.0073, 0.0000), (0.0292, 0.0717, 0.0000, 0.0141, 0.0337, 0.0000)
race_means, race_std = (0.0769, 0.0549, 0.0602, 0.0713, 0.1563, 0.0253), (0.2188, 0.1948, 0.1818, 0.2177, 0.2990, 0.1213)
race_unknown_means, race_unknown_std = (0.0084, 0.0213, 0.0042, 0.0009, 0.0211, 0.0000), (0.0682, 0.1239, 0.0185, 0.0121, 0.1097, 0.0000)
household_income_means, household_income_std = (0.0134, 0.0294, 0.0028, 0.0104, 0.0295, 0.0231), (0.0600, 0.0935, 0.0190, 0.0549, 0.0919, 0.0551)
household_income_unknown_means, household_income_unknown_std = (0.0016, 0.0036, 0.0012, 0.0015, 0.0021, 0.0000), (0.0136, 0.0202, 0.0111, 0.0155, 0.0139, 0.0000)

ind = np.arange(len(insurance_means))  # the x locations for the groups
width = 0.12  # the width of the bars
offsets = np.array([
    -2.5, -1.5, -0.5,
     0.5,  1.5,  2.5
]) * width

fig, ax = plt.subplots()
ax.bar(ind + offsets[0], insurance_means, width, yerr=insurance_std,
                label='Insurance')
ax.bar(ind + offsets[1], insurance_unknown_means, width, yerr=insurance_unknown_std,
                label='Insurance (Unknown Option)')
ax.bar(ind + offsets[2], race_means, width, yerr=race_std,
                label='Race')
ax.bar(ind + offsets[3], race_unknown_means, width, yerr=race_unknown_std,
                label='Race (Unknown Option)')
ax.bar(ind + offsets[4], household_income_means, width, yerr=household_income_std,
                label='Household Income')
ax.bar(ind + offsets[5], household_income_unknown_means, width, yerr=household_income_unknown_std,
                label='Household Income (Unknown Option)')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_title('Percentage of Relevant Sentences Based on Question')
ax.set_xticks(ind)
ax.set_xticklabels(('Overall', 'MMLU', 'Jama', 'Medxpert', 'Medbullets', 'Q-Pain'))
ax.legend()

fig.tight_layout()
plt.show()